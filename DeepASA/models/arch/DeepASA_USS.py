import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from packaging.version import parse as V
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
from models.arch.mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import loralib as lora


class DeepASA_USS(nn.Module):
    def __init__(self, n_srcs=4, n_mics=4, n_cls=2, n_layers=4, att_dim=64, hidden_dim=256, n_head=4, emb_dim=64, emb_ks=4, emb_hs=1, dropout=0.1, eps=1.0e-5):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_mics = n_mics
        self.n_cls = n_cls
        self.emb_dim = emb_dim

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        # encoder
        self.up_conv = nn.Sequential(
            nn.Conv2d(2*n_mics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        # feature extraction
        self.blocks = nn.ModuleList([])
        for idx in range(n_layers):
            self.blocks.append(DeFTANblock(idx, emb_dim, emb_ks, emb_hs, hidden_dim, n_head, dropout, eps))
        # object extraction
        self.object_conv = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim*(self.n_srcs+1), ks, padding=padding),
            nn.Mish(inplace=False),
            nn.GroupNorm(1, emb_dim*(self.n_srcs+1))
        )
        # separation
        self.audio_conv = nn.Conv2d(emb_dim, 2, ks, padding=padding)
        self.noise_conv_ = nn.Conv2d(emb_dim, 2, ks, padding=padding)

    def _freeze_module(self, modules):
        """
        'Module' means the freezed modules
        """
        for name, param in self.named_parameters():
            for fn in modules:
                if fn in name:
                    param.requires_grad = False

    def forward(self, input):
        # Encoding (STFT)
        input = input.permute(0, 3, 2, 1).contiguous()                              # [B, 2*M, T, F]
        B, _, T, F = input.shape
        batch = self.up_conv(input)                                                    # [B, C, T, F]
        # Feature Extraction
        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)                                          # [B, C, T, F]
        batch = self.object_conv(batch).reshape(B, self.n_srcs+1, self.emb_dim, T, F)
        noise_out = self.noise_conv_(batch[:, -1, :, :, :].reshape(B, self.emb_dim, T, F))  # [B, 2*M, T, F]
        batch = batch[:, :self.n_srcs, :, :, :].reshape(-1, self.emb_dim, T, F)     # [BS, C, T, F]
        # Decoding (Multi-Task)
        sep_out = self.audio_conv(batch).reshape(B, 2*self.n_srcs, T, F).permute(0, 3, 2, 1).contiguous()  # [B, F, T, 2*S]
        noise_out = noise_out.permute(0, 3, 2, 1).contiguous()  # [B, F, T, 2*M]
        
        return sep_out, noise_out
    

class ObjectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, use_lora=False):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.Conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.Mish = nn.Mish(inplace=False)

    def forward(self, x):
        return self.Conv1(x) * self.Mish(self.Conv2(x))


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, use_lora):
        dim_head = dim
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)

        self.dropout = dropout

        if not project_out:
            self.to_out = nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
            # self.to_out =  nn.Linear(inner_dim, dim)
            # self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h = self.heads)
        # FlashAttention
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=self.dropout)
        out = rearrange(out.transpose(1, 2), 'b h n d -> b n (h d)')
        # return self.out_dropout(self.to_out(out.to(torch.float32)))
        return self.to_out(out.to(torch.float32))


class BiMamba(nn.Module):
    def __init__(self, in_channels, n_layer=1, bidirectional=True):
        super(BiMamba, self).__init__()
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mixer_cls=partial(Mamba2Simple, layer_idx=i, d_state=16, d_conv=4, expand=4, headdim=16),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    mlp_cls=nn.Identity,
                    fused_add_norm=False,
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                        Block(
                        in_channels,
                        mixer_cls=partial(Mamba2Simple, layer_idx=i, d_state=16, d_conv=4, expand=4, headdim=16),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        mlp_cls=nn.Identity,
                        fused_add_norm=False,
                    )
                )
        self.apply(partial(_init_weights, n_layer=n_layer))
        self.FC = nn.Linear(2*in_channels, 2*in_channels)

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)
        output = self.FC(residual)
        return output


class MambaFFN(nn.Module):
    def __init__(self, dim, hidden_dim, idx, dropout):
        super().__init__()
        self.net = nn.Sequential(
            BiMamba(dim, 1),
            nn.Mish(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, idx, dropout, use_lora=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Mish(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_lora=False):
        super().__init__()
        self.Conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.Conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.Mish = nn.Mish()

    def forward(self, x):
        return self.Conv1(x) * self.Mish(self.Conv2(x))


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class LinearGroup(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True) -> None:
        super(LinearGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = Parameter(torch.empty((num_groups, out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same as linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """shape [..., group, feature]"""
        x = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, bias={True if self.bias is not None else False}"


class DeFTANblock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(self, idx, emb_dim, emb_ks, emb_hs, hidden_dim, n_head, dropout, eps):
        super().__init__()
        in_channels = emb_dim
        # Frequency Module
        self.intra_norm = LayerNormalization4D(emb_dim, eps)
        self.intra_inv = GatedConv(in_channels, emb_dim, use_lora=False)
        self.intra_att = PreNorm(emb_dim, Attention(emb_dim, n_head, dropout, use_lora=False))
        self.intra_ffw = PreNorm(emb_dim, FeedForward(emb_dim, hidden_dim, idx, dropout))    
        # Time Module
        self.inter_norm = LayerNormalization4D(emb_dim, eps)
        self.inter_inv = GatedConv(in_channels, emb_dim)
        self.inter_att = PreNorm(emb_dim, Attention(emb_dim, n_head, dropout, use_lora=False))
        self.inter_ffw = PreNorm(emb_dim, MambaFFN(emb_dim, hidden_dim, idx, dropout))

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # F-transformer
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)  # [BT, C, Q]
        intra_rnn = self.intra_inv(intra_rnn) + intra_rnn   # [BT, C, -1]

        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C]
        intra_rnn = self.intra_att(intra_rnn) + intra_rnn
        intra_rnn = self.intra_ffw(intra_rnn) + intra_rnn
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]

        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]

        # T-transformer
        inter_rnn = intra_rnn
        inter_rnn = self.inter_norm(inter_rnn)  # [B, C, T, F]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)  # [BF, C, T]
        inter_rnn = self.inter_inv(inter_rnn) + inter_rnn   # [BF, C, -1]

        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C]
        inter_rnn = self.inter_att(inter_rnn) + inter_rnn
        inter_rnn = self.inter_ffw(inter_rnn) + inter_rnn
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]

        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + intra_rnn  # [B, C, T, Q]

        return inter_rnn


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat