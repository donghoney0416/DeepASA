# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from IPython import embed
from transformers import ASTModel
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from functools import partial
from models.arch.mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm
# from models.utils.vision_transformer import SimpleViT
from models.arch.ATST import ATST


class DoANet(nn.Module):
    def __init__(self, time_dim=251, freq_dim=257, input_channels=64, pool_size=[[8,5],[8,1]], pool_time=True,  n_cnn_filters=64, rnn_size=128, n_rnn=2,fc_size=128, dropout_perc=0., verbose=False):
        super(DoANet, self).__init__()
        self.verbose = verbose
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        if pool_time:
            self.time_pooled_size = int(time_dim / np.prod(np.array(pool_size), axis=0)[-1])
        else:
            self.time_pooled_size = time_dim
        #building CNN feature extractor
        conv_layers = []
        in_chans = input_channels
        for p in pool_size:
            curr_chans = n_cnn_filters
            if pool_time:
                pool = [p[0],p[1]]
            else:
                pool = [p[0],1]
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_chans, out_channels=curr_chans, kernel_size=3, stride=1, padding=1),  #padding 1 = same with kernel = 3
                    nn.GroupNorm(1, n_cnn_filters, 1e-5),
                    nn.Mish(inplace=True),
                    nn.MaxPool2d(pool),
                    nn.Dropout(dropout_perc)))
            in_chans = curr_chans
        self.cnn = nn.Sequential(*conv_layers)
        self.rnn = nn.GRU(320, rnn_size, num_layers=n_rnn, batch_first=True, bidirectional=True, dropout=dropout_perc)
        self.doa = nn.Sequential(
                    nn.Linear(256, fc_size),
                    nn.Dropout(dropout_perc),
                    nn.Linear(fc_size, 3),
                    nn.Tanh())
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0,3,1,2) #[batch, time, channels, freq]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, h = self.rnn(x)
        doa = self.doa(x)
        return doa
    
    
    class AttentiveStatsPool(nn.Module):
    """
    x의 shape: (batch, frames, dim)
    output shape: (batch, 2*dim)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(AttentiveStatsPool, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.linear3 = nn.Linear(2*input_dim, input_dim)
        
    def forward(self, x):
        e = self.linear2(torch.tanh(self.linear1(x)))  # (B, T, 1)
        alpha = F.softmax(e, dim=1)  # (B, T, 1)
        mu = torch.sum(alpha * x, dim=1)  # (B, D)
        second_moment = torch.sum(alpha * x * x, dim=1)  # (B, D)
        sigma = torch.sqrt(torch.clamp(second_moment - mu * mu, min=1e-5))  # (B, D)
        pooled = torch.cat([mu, sigma], dim=1)  # (B, 2D)
        return self.linear3(pooled)


class ClassNet(nn.Module):
    def __init__(self, time_dim=201, freq_dim=321, input_channels=64, n_cls=13, pool_size_t=[[2,5],[2,1],[2,1],[2,1],[2,1],[2,1],[1,1]], pool_size_f=[[4,5],[4,4]], pool_time=True, n_cnn_filters=64, rnn_size=128, n_rnn=2, fc_size=128, dropout_perc=0., verbose=False):
        super(ClassNet, self).__init__()
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        if pool_time:
            self.time_pooled_size = int(time_dim / np.prod(np.array(pool_size_t), axis=0)[-1])
        else:
            self.time_pooled_size = time_dim
        # building T-CNN feature extractor
        conv_layers_t = []
        in_chans = input_channels
        for p in pool_size_t:
            curr_chans = n_cnn_filters
            if pool_time:
                pool = [p[0],p[1]]
            else:
                pool = [p[0],1]
            conv_layers_t.append(
                nn.Sequential(
                    nn.Conv2d(in_chans, out_channels=curr_chans, kernel_size=3, stride=1, padding=1),  #padding 1 = same with kernel = 3
                    nn.GroupNorm(1, n_cnn_filters, 1e-5),
                    nn.Mish(inplace=True),
                    nn.MaxPool2d(pool),
                    nn.Dropout(dropout_perc)))
            in_chans = curr_chans
        self.cnn_t = nn.Sequential(*conv_layers_t)
        # building ATST feature extractor
        self.atst = ATST()
        self.adaptivepooling1d = nn.AdaptiveAvgPool1d(40)
        self.downsize = nn.Linear(768, 320)
        #building F-CNN feature extractor
        conv_layers_f = []
        in_chans = input_channels
        for p in pool_size_f:
            curr_chans = n_cnn_filters
            pool = [p[0],p[1]]
            conv_layers_f.append(
                nn.Sequential(
                    nn.Conv2d(in_chans, out_channels=curr_chans, kernel_size=3, stride=1, padding=1),  #padding 1 = same with kernel = 3
                    nn.GroupNorm(1, n_cnn_filters, 1e-5),
                    nn.Mish(inplace=True),
                    nn.MaxPool2d(pool),
                    nn.Dropout(dropout_perc)))
            in_chans = curr_chans
        self.cnn = nn.Sequential(*conv_layers_f)
        # SED
        self.Linear_t = nn.Linear(640, 320)
        self.Linear = nn.Linear(640, 320)
        self.rnn_t = nn.GRU(320, rnn_size, num_layers=n_rnn, batch_first=True, bidirectional=True, dropout=dropout_perc)
        self.freq_gru = nn.GRU(320, rnn_size, num_layers=n_rnn, batch_first=True, bidirectional=True, dropout=dropout_perc)
        self.sed = nn.Sequential(
                    nn.Linear(256, fc_size),
                    nn.Dropout(dropout_perc),
                    nn.Linear(fc_size, n_cls))
        self.sed_t = nn.Sequential(
                    nn.Linear(256, fc_size),
                    nn.Dropout(dropout_perc),
                    nn.Linear(fc_size, n_cls))
        self.softmax = nn.Softmax(dim=-1)         # softmax on class dimension
        self.asp = AttentiveStatsPool(n_cls)
        self.onoff = nn.Sequential(
                    nn.Linear(256, fc_size),
                    nn.Dropout(dropout_perc),
                    nn.Linear(fc_size, 1))

    def forward(self, x):
        input = x
        # CNN path
        x1 = self.cnn_t(input)
        x1 = x1.permute(0,3,1,2) #[batch, time, channels, freq]
        x1 = x1.reshape(x1.shape[0], x1.shape[1], -1)
        # ATST path
        x2 = self.atst(input)
        x2 = self.downsize(self.adaptivepooling1d(x2).permute(0, 2, 1).contiguous())    # [batch, time', 320]
        # Classifier
        x,h = self.rnn_t(self.Linear_t(torch.cat([x1, x2], dim=-1)))                                                             # [batch, time, 320]
        # F-CNN & F-GRU path
        x3 = self.cnn(input)     # [batch, channels, freq, time]
        B, C, F, T = x3.shape
        x3 = x3.permute(0, 2, 1, 3).contiguous().reshape(B, F, C*T)
        x3 = self.Linear(x3)
        x3 = self.freq_gru(x3)[0]
        x3 = self.sed(x3.mean(dim=1))
        ## cross entropy loss
        strong = self.sed_t(x)
        weak = self.asp(strong)
        onoff = self.onoff(x)
        return weak + x3, torch.softmax(strong, dim=-1), onoff