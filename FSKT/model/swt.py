import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSWT(nn.Module):
    def __init__(self, d_model, n_levels=2, wavelet='haar', dropout=0.1, learnable=False):
        super().__init__()

        self.d_model = d_model
        self.n_levels = n_levels
        self.learnable = learnable

        h_low, h_high = self._get_wavelet_filters(wavelet)
        self.kernel_size = len(h_low)

        self.low_convs = nn.ModuleList()
        self.high_convs = nn.ModuleList()

        for level in range(n_levels):
            dilation = 2 ** level
            low_conv = nn.Conv1d(
                d_model, d_model,
                kernel_size=self.kernel_size,
                dilation=dilation,
                padding=0, 
                groups=d_model, 
                bias=False
            )

            high_conv = nn.Conv1d(
                d_model, d_model,
                kernel_size=self.kernel_size,
                dilation=dilation,
                padding=0,  
                groups=d_model, 
                bias=False
            )

            with torch.no_grad():
                low_conv.weight[:, 0, :] = h_low.view(1, -1)  
                high_conv.weight[:, 0, :] = h_high.view(1, -1)

            if not learnable:
                low_conv.weight.requires_grad = False
                high_conv.weight.requires_grad = False

            self.low_convs.append(low_conv)
            self.high_convs.append(high_conv)

        self.fuse_g = nn.Linear(d_model , d_model)
        self.fuse_t = nn.Linear(d_model * n_levels, d_model)

        self.norm_g = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        fan_in = self.d_model * self.n_levels
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.fuse_t.weight, mean=0.0, std=std)
        nn.init.zeros_(self.fuse_t.bias)

    def _get_wavelet_filters(self, wavelet):
        if wavelet == 'haar':
            h_low = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)])
            h_high = torch.tensor([1/math.sqrt(2), -1/math.sqrt(2)])
        else:
            raise ValueError(f"Unsupported wavelet: {wavelet}")

        return h_low, h_high

    def forward(self, h):
        B, T, D = h.shape

        h_conv = h.permute(0, 2, 1)

        g_levels = []
        t_levels = []

        current_input = h_conv
        for level in range(self.n_levels):
            causal_padding = (self.kernel_size - 1) * (2 ** level)

            if causal_padding > 0:
                padded_input = F.pad(current_input, (causal_padding, 0), mode='constant', value=0)
            else:
                padded_input = current_input

            g_conv = self.low_convs[level](padded_input)
            t_conv = self.high_convs[level](padded_input)

            g_level = g_conv.permute(0, 2, 1)
            t_level = t_conv.permute(0, 2, 1)

            g_levels.append(g_level)
            t_levels.append(t_level)

            current_input = g_conv

        t_cat = torch.cat(t_levels, dim=-1)
        g = self.fuse_g(g_levels[-1])
        t = self.fuse_t(t_cat) 

      
        g = self.norm_g(self.dropout(g))
        t = self.norm_t(self.dropout(t))

        return g, t