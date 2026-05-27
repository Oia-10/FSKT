import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import Embedding
from .mattention import TransformerBlock
from .swt import CausalSWT
from .scd_parallel import SCDNetParallel


class FSKT(nn.Module):
    def __init__(self, num_q, num_c, embed_dim, hidden_dim, Q_matrix,
                 dropout=0.2, n_blocks=2, n_heads=4, d_ff=None,
                n_levels=2, l2=1e-5, kq_same=True, add=True):
        super().__init__()
        self.l2 = l2
        self.embed_dim = embed_dim
        self.num_c = num_c
        self.add = add
        d_ff = d_ff or embed_dim * 2

        self.register_buffer('Q_matrix', Q_matrix)

        self.embedding = Embedding(num_q, num_c, embed_dim, self.Q_matrix)

        self.qa_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, d_ff, dropout, kq_same)
            for _ in range(n_blocks)
        ])

        self.swt = CausalSWT(
            d_model=embed_dim,
            n_levels=n_levels,
            wavelet='haar',
            dropout=dropout,
            learnable=False
        )

        
        self.scd_net = SCDNetParallel(
            d_model=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.out1 = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )        
        self.out2 = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )

        

    def forward(self, q, r, mask=None):
        cqa_embed, cq_embed,q_embed, c_embed, r_embed, q_diff = self.embedding(q, r)

        y = cqa_embed
        x = cq_embed
        for block in self.qa_blocks:
            x = block(x, x, y, peek_cur=False)
        h_pre = x[:, 1:]  # [B, L-1, d]

        g, t = self.swt(h_pre)
        extra = {
            'g': g,
            't': t,
            'reg_loss': self.embedding.get_reg_loss(q) * self.l2
        }

        stages, scd_loss = self.scd_net(
            g=g,
            t=t,
            q_embed = q_embed[:, :-1],
            r_embed = r_embed[:, :-1],
            c_embed = c_embed[:, :-1],
            q_embed_next = cq_embed[:, 1:, :],  
            mask=mask[:, :-1] if mask is not None else None
        )
        extra['scd_loss'] = scd_loss
        extra['stages'] = stages

        _, _, AC, AE = stages

        if self.add:
            concat_q1 = torch.cat([AC+g, cq_embed[:, 1:]], dim=-1)
            concat_q2 = torch.cat([AE+t, q_embed[:, 1:]], dim=-1)
        else:
            concat_q1 = torch.cat([AC, cq_embed[:, 1:]], dim=-1)
            concat_q2 = torch.cat([AE, q_embed[:, 1:]], dim=-1)


        output1 = self.out1(concat_q1).squeeze(-1)
        pred_ac = torch.sigmoid(output1)

        output2 = self.out2(concat_q2).squeeze(-1)
        pred_ae = torch.sigmoid(output2)

        pred = (pred_ac + pred_ae) / 2


        extra['pred_ac'] = pred_ac
        extra['pred_ae'] = pred_ae


        return pred, extra
