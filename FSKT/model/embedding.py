import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    - q: question
    - c: concept 
    - r: response 
    """

    def __init__(self, num_q, num_c, embed_dim, Q_matrix, use_diff=True):
        super().__init__()
        self.use_diff = use_diff
        self.num_c = num_c
        self.num_q = num_q

        self.c_embed = nn.Embedding(num_c + 1, embed_dim, padding_idx=0)
        self.r_embed = nn.Embedding(2, embed_dim)
        self.register_buffer('Q', Q_matrix)

        if use_diff:
            self.q_diff_param = nn.Embedding(num_q + 1, 1, padding_idx=0)
            self.c_embed_diff = nn.Embedding(num_c + 1, embed_dim, padding_idx=0)
            self.r_embed_diff = nn.Embedding(2, embed_dim)

        self._init_weights()

    def _init_weights(self):
        if self.use_diff:
            nn.init.constant_(self.q_diff_param.weight, 0.)

    def forward(self, q, r):

        c_embed = self._get_concept_emb(q)  # [B, L, d]
        r_embed = self.r_embed(r)  # [B, L, d]

        ca_embed = r_embed + c_embed  # [B, L, d]

        q_diff = None
        if self.use_diff:
            q_diff = self.q_diff_param(q)  # [B, L, 1]
            c_embed_diff = self._get_concept_diff_emb(q)  # [B, L, d]
            q_embed = q_diff * c_embed_diff
            cq_embed = c_embed + q_diff * c_embed_diff
            r_embed_diff = self.r_embed_diff(r)  # [B, L, d]
            cqa_embed = ca_embed + q_diff * (r_embed_diff + c_embed_diff)
        return cqa_embed, cq_embed, q_embed,c_embed, r_embed, q_diff

    def _get_concept_emb(self, q):
        q_mask = self.Q[q]  # [B, L, num_c+1]
        c_emb_sum = torch.matmul(q_mask, self.c_embed.weight)  # [B, L, d]
        c_count = q_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, L, 1]
        return c_emb_sum / c_count

    def _get_concept_diff_emb(self, q):
        q_mask = self.Q[q]  # [B, L, num_c+1]
        c_diff_sum = torch.matmul(q_mask, self.c_embed_diff.weight)  # [B, L, d]
        c_count = q_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, L, 1]
        return c_diff_sum / c_count

    def get_reg_loss(self, q):
        if self.use_diff:
            q_diff = self.q_diff_param(q)  # [B, L, 1]
            return (q_diff ** 2).sum()  
        return 0.0

