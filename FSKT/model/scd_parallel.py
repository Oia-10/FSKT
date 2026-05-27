
import torch
import torch.nn as nn
import torch.nn.functional as F


class CGPU(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        hidden_dim = hidden_dim or d_model

        self.W_a = nn.Linear(d_model, d_model, bias=False)
        self.W_c = nn.Linear(d_model, d_model, bias=False)
        self.query_norm = nn.LayerNorm(d_model)

        self.W_h = nn.Linear(d_model, d_model, bias=False)
        self.W_q = nn.Linear(d_model, d_model, bias=False)

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid()
        )

        self.extract_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )

        self.W_proj = nn.Linear(d_model, d_model, bias=False)
        self.res_norm = nn.LayerNorm(d_model)

        self.apply(self._init_weights)

        gate_mlp_layers = [m for m in self.gate_mlp.modules() if isinstance(m, nn.Linear)]
        if len(gate_mlp_layers) > 0:
            gate_mlp_layers[-1].bias.data.fill_(1.5)  

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, h, anchor, context):
        q = self.query_norm(self.W_a(anchor) + self.W_c(context))
        gate = self.gate_mlp(self.W_h(h) * self.W_q(q))
        f_raw = self.extract_mlp(h * gate)
        u = F.normalize(f_raw, p=2, dim=-1)

        h_mag = torch.norm(h, p=2, dim=-1, keepdim=True) + 1e-6
        h_projected = self.W_proj(h)
        ratio = torch.sigmoid((h_projected * u).sum(dim=-1, keepdim=True))
        alpha = ratio * h_mag
        state = alpha * u

        # h_res = self.res_norm(h - state)
        h_res = h - state
        return state, h_res


class SCDNetParallel(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        hidden_dim = hidden_dim or d_model

        self.cgpu_CE = CGPU(d_model, hidden_dim, dropout)
        self.cgpu_RO = CGPU(d_model, hidden_dim, dropout)
        self.cgpu_AC = CGPU(d_model, hidden_dim, dropout)
        self.cgpu_AE = CGPU(d_model, hidden_dim, dropout)

    def forward(self, g, t, q_embed, r_embed, c_embed, q_embed_next, mask=None):
        B, T, D = g.shape

        t_prev = F.pad(t[:, :-1, :], (0, 0, 1, 0), value=0)  # [B, T, D]

        CE, g_res = self.cgpu_CE(g, anchor=q_embed, context=t_prev)
        RO, t_res = self.cgpu_RO(t, anchor=r_embed, context=CE)
        AC, g_res2 = self.cgpu_AC(g_res, anchor=c_embed, context=RO)
        AE, t_res2 = self.cgpu_AE(t_res, anchor=q_embed_next, context=AC)

        stacked = {'CE': CE, 'RO': RO, 'AC': AC, 'AE': AE,
                   'g_res': g_res, 't_res': t_res, 'g_res2': g_res2, 't_res2': t_res2}
        loss_dict = self.compute_sufficiency_loss(
            stacked, q_embed, r_embed, c_embed, q_embed_next, mask=mask)

        return (CE, RO, AC, AE), loss_dict

    def compute_sufficiency_loss(self, stacked_outputs,
                                 q_anchor, r_anchor, c_anchor, q_next_anchor,
                                 mask=None):
        loss_dict = {}

        def get_loss(key, h_res_key, anchor):
            return self.sufficiency_loss(
                stacked_outputs[key],
                stacked_outputs[h_res_key],
                anchor,
                mask
            )

        loss_dict['suff_CE'] = get_loss('CE', 'g_res', q_anchor)
        loss_dict['suff_RO'] = get_loss('RO', 't_res', r_anchor)
        loss_dict['suff_AC'] = get_loss('AC', 'g_res2', c_anchor)
        loss_dict['suff_AE'] = get_loss('AE', 't_res2', q_next_anchor)

        loss_dict['suff_total'] = sum(loss_dict.values())
        return loss_dict

    def sufficiency_loss(self, f, h_res, anchor, mask=None):
        f_norm = F.normalize(f, p=2, dim=-1)
        h_res_norm = F.normalize(h_res.detach(), p=2, dim=-1)
        anchor_norm = F.normalize(anchor, p=2, dim=-1)

        sim_f_a = (f_norm * anchor_norm).sum(dim=-1)
        sim_res_a = (h_res_norm * anchor_norm).sum(dim=-1)

        # Log-Sigmoid Loss
        diff = sim_f_a - sim_res_a
        loss_map = F.softplus(-diff)

        # Masking
        if mask is not None:
            loss_map = loss_map * mask
            return loss_map.sum() / (mask.sum() + 1e-9)
        else:
            return loss_map.mean()
