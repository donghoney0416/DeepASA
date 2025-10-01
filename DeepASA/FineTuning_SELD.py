from models.utils.base_cli import BaseCLI
# import BaseCLI at the beginning

import os
from typing import *
import wandb

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jsonargparse import lazy_instance
from packaging.version import Version
from torch import Tensor
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio.pit import _find_best_perm_by_linear_sum_assignment, _find_best_perm_by_exhaustive_method
from torchmetrics.functional.audio import pit_permutate
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
# from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from visualize_SELD import *
from scipy.io.wavfile import write
from models.utils.SEBB import get_sebb_mask, post_processing
from ray import tune
import pandas as pd
import scipy.io as sio

import models.utils.general_steps as GS
import math
from torch.nn.utils import clip_grad_norm_
from models.io.loss import *
from models.io.norm import Norm
from models.io.stft import STFT, GaborSTFT
from models.utils.metrics import (cal_metrics_functional, recover_scale)
from models.utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback
from models.utils.SELD_evaluation_metrics import *
import itertools
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model


def calculate_loss(self, cls_out, sed_out, doa_out, onoff_out, cls_label, sed_label, doa_label):
    B, _ = cls_out.shape
    loss = torch.zeros(B, requires_grad=True, device=cls_out.device)
    # 1) Classification loss
    if cls_out != None:
        cls_loss_ = self.cls_loss(cls_out, cls_label)
        loss = loss + self.loss_weight['cls_loss'] * cls_loss_
    # 2) SED loss
    if sed_out != None:
        sed_loss_ = self.sed_loss(sed_out, sed_label)
        loss = loss + self.loss_weight['sed_loss'] * sed_loss_
    # 3) DoA loss
    doa_loss_ = self.doa_loss(doa_out, doa_label)
    cos_loss_ = cossim_loss(doa_out, doa_label, keep_batch=True)
    loss = loss + self.loss_weight['doa_loss'] * doa_loss_ + self.loss_weight['cos_loss'] * cos_loss_
    # 4) On/offset loss
    if onoff_out != None:
        onoff_gt = (torch.norm(doa_label, dim=-1) > 0.5).float()
        onoff_loss_ = self.onoff_loss(onoff_out, onoff_gt)
        loss = loss + self.loss_weight['onoff_loss'] * onoff_loss_
    loss_log = {
        'cls_loss': cls_loss_ if cls_out != None else None,
        'doa_loss': doa_loss_ if doa_out != None else None,
        'cos_loss': cos_loss_ if doa_out != None else None,
        'onoff_loss': onoff_loss_ if onoff_out != None else None
    }
    return loss, None, loss_log


def loss_fn(self, cls_out, sed_out, doa_out, onoff_out, cls_label, sed_label, doa_label, mode, weight):
    B, S, C = cls_out.shape
    B, _, T, _ = doa_out.shape

    perms = torch.tensor(list(itertools.permutations(range(S))), device=cls_out.device)
    perms_num = perms.shape[0]

    pcls_out = torch.index_select(cls_out, dim=1, index=perms.reshape(-1)).reshape(B * perms_num  * S, *cls_out.shape[2:])
    psed_out = torch.index_select(sed_out, dim=1, index=perms.reshape(-1)).reshape(B * perms_num * S, *sed_out.shape[2:])
    pdoa_out = torch.index_select(doa_out, dim=1, index=perms.reshape(-1)).reshape(B * perms_num * S, *doa_out.shape[2:])
    ponoff_out = torch.index_select(onoff_out, dim=1, index=perms.reshape(-1)).reshape(B * perms_num * S, *onoff_out.shape[2:])

    pcls_label = cls_label.repeat_interleave(repeats=perms_num, dim=0).reshape(B * perms_num * S, *cls_label.shape[2:])
    psed_label = sed_label.repeat_interleave(repeats=perms_num, dim=0).reshape(B * perms_num * S, *sed_label.shape[2:])
    pdoa_label = doa_label.repeat_interleave(repeats=perms_num, dim=0).reshape(B * perms_num * S, *doa_label.shape[2:])

    matric_of_ps, _, loss_log = calculate_loss(self, pcls_out, psed_out, pdoa_out, ponoff_out, pcls_label, psed_label, pdoa_label)
    matric_of_ps = torch.mean(matric_of_ps.reshape(B, len(perms), -1), dim=-1)
    matric_of_ps = torch.where(torch.isnan(matric_of_ps), torch.tensor(float('inf'), device=cls_out.device), matric_of_ps)
    loss_log = {k: torch.mean(v.reshape(B, len(perms), -1), dim=-1) for k, v in loss_log.items()}

    best_matric, best_indexes = torch.min(matric_of_ps, dim=1)
    best_indexes = best_indexes.detach()
    loss_log = {k: v[torch.arange(B), best_indexes] for k, v in loss_log.items()}
    best_perm = perms[best_indexes]

    return best_matric.mean(), best_perm, None, loss_log


class TrainModule(pl.LightningModule):
    """Network Lightning Module, which controls the training, testing, and inference of given arch and io
    """
    name: str  # used by CLI for creating logging dir
    import_path: str = 'SharedTrainer.TrainModule'

    def __init__(
        self,
        arch: nn.Module,
        channels: List[int],
        ref_channel: int,
        # stft: STFT = STFT(n_fft=400, n_hop=160, win_len=400),
        stft: GaborSTFT = GaborSTFT(n_fft=400, n_hop=320, win_len=400, num_frames=201),
        norm: Norm = Norm(mode='utterance'),
        loss: Loss = Loss(loss_func=neg_sa_sdr, pit=True),
        loss_weight: Optional[Dict[str, float]] = None,
        optimizer: Tuple[str, Dict[str, Any]] = ("Adam", {
            "lr": 0.001
        }),
        lr_scheduler: Optional[Tuple[str, Dict[str, Any]]] = ('ReduceLROnPlateau', {
            'mode': 'min',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-4
        }),
        metrics: List[str] = ['SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ', 'eSTOI'],
        val_metric: str = 'loss',
        write_examples: int = 200,
        ensemble: Union[int, str, List[str], Literal[None]] = None,
        compile: bool = False,
        exp_name: str = "exp",
        pre_trained_ckpt: str = '',
        wandb_dict: Dict[str, Any] = None,
        use_lora: Optional[List[LoraConfig]] = [],
    ):
        """
        Args:
            exp_name: set exp_name to notag when debug things. Defaults to "exp".
            metrics: metrics used at test time. Defaults to ['SNR', 'SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ'].
            write_examples: write how many examples at test.
        """
        super().__init__()
        args = locals().copy()  # capture the parameters passed to this function or their edited values

        self.channels = channels
        self.ref_channel = ref_channel
        self.stft = stft
        self.norm = norm
        # loss function
        self.sep_loss = Loss(loss_func=neg_sa_sdr, pit=True,)
        self.cls_loss = AudioClassificationLoss(keep_batch=True)
        self.sed_loss = MSELoss(keep_batch=True)
        self.doa_loss = MSELoss(keep_batch=True)
        self.onoff_loss = BCEWithLogitsLoss(keep_batch=True)

        # Analysis tool
        self.er_values = []
        self.le_values = []
        self.cls_out = []
        self.cls_label = []
        self.features = []
        self.f_features = []
        self.onoff_out = []
        self.sed_label = []
        self.doa_out = []
        self.doa_label = []

        self.val_wandb_log = {}

        self.val_cpu_metric_input = []
        self.norm_if_exceed_1 = True
        self.name = type(arch).__name__

        for lora_config in use_lora:
            arch = get_peft_model(arch, lora_config)

        for n, p in arch.named_parameters():
            if 'base_layer' not in n:
                p.requires_grad = True
            # print(f'{n} {p.requires_grad}')

        # save other parameters to self
        for k, v in args.items():
            if k == 'self' or k == '__class__' or hasattr(self, k):
                continue
            setattr(self, k, v)

    def on_train_start(self):
        """Called by PytorchLightning automatically at the start of training"""
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {lr}")

        GS.on_train_start(self=self, exp_name=self.exp_name, model_name=self.name, num_chns=max(self.channels) + 1, nfft=self.stft.n_fft, model_class_path=self.import_path)

    def process_output(self, output, B, C, F, T, x_shape, norm, stft, norm_paras, stft_paras, istft):
        output = output.reshape(B, F, T, C, -1).permute(0, 3, 1, 2, 4).contiguous().reshape(B * C, F, T, -1)
        if not torch.is_complex(output):
            output = torch.view_as_complex(output.float().reshape(B * C, F, T, -1, 2))
        output = output.permute(0, 3, 1, 2).reshape(B, C, -1, F, T)  # [B, M, Spk, F, T]
        output_hat = torch.zeros(B, C, output.shape[2], x_shape[-1], device=output.device)
        for i in range(C):
            output_hat[:, i] = stft.istft(norm.inorm(output[:, i], norm_paras), stft_paras) if istft else torch.view_as_real(output[:, i])
        return output_hat

    def forward(self, x, istft=True):
        """
        Args:
            x: [B,M,N]

        Returns:
            yr_hat: [B,M,N], yr_ane_hat: [B,M,N], yr_rev_hat: [B,M,N], cls_out: [B,S,L,C], sed_out: [B,S,L,C], doa_out: [B,S,L,3], onoff_out: [B,S,L]
        """
        # obtain STFT X
        X, stft_paras = self.stft.stft(x[:, self.channels])  # [B,CM,F,T], complex
        B, C, F, T = X.shape
        X, norm_paras = self.norm.norm(X, ref_channel=self.channels.index(self.ref_channel))
        X = X.permute(0, 2, 3, 1)  # B,F,T,M; complex
        X = torch.view_as_real(X).reshape(B, F, T, -1)  # B,F,T,2M

        sep_out, ane_out, rev_out, _, cls_out, sed_out, doa_out, onoff_out = self.arch(X)                    # [B, F, T, 2*M*S]
        order = [10, 1, 3, 12, 5, 0, 6, 9, 4, 2, 8, 11, 7]
        # order = [10, 8, 12, 0, 6, 5, 4, 7, 9, 2, 1, 11, 3]
        cls_out = cls_out[..., order]
        sed_out = sed_out[..., order]
        ## MIMO separation
        if sep_out != None:
            yr_hat = self.process_output(sep_out, B, C, F, T, x.shape, self.norm, self.stft, norm_paras, stft_paras, istft)
            yr_ane_hat, yr_rev_hat = None, None
        elif ane_out != None:
            yr_ane_hat = self.process_output(ane_out, B, C, F, T, x.shape, self.norm, self.stft, norm_paras, stft_paras, istft)
            yr_rev_hat = self.process_output(rev_out, B, C, F, T, x.shape, self.norm, self.stft, norm_paras, stft_paras, istft)
            yr_hat = None

        return yr_hat, yr_ane_hat, yr_rev_hat, cls_out, sed_out, doa_out, onoff_out

    def training_step(self, batch, batch_idx):
        """training step on self.device, called automaticly by PytorchLightning"""
        x, cls_label, sed_label, doa_label, num_spk, paras = batch  # x: [B,C,T], ys: [B,Spk,C,T]
        # x = x/(torch.max(torch.abs(x), dim=-1).values.unsqueeze(-1)+1e-8)  # normalize the input
        scaling_factor = torch.amax(torch.abs(x), dim=(1, 2), keepdim=True)
        x = x/(scaling_factor+1e-8)  # normalize the input

        y_hat, y_ane_hat, y_rev_hat, cls_out, sed_out, doa_out, onoff_out = self.forward(x)
        # float32 loss calculation
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            loss, perms, _, loss_log = loss_fn(self, cls_out, sed_out, doa_out, onoff_out, cls_label, sed_label, doa_label, mode='train', weight=self.loss_weight)

        # Wandb logging
        # wandb_log = {'train/loss': loss}
        # for k, v in loss_log.items():
        #     if v is not None:
        #         wandb_log[f'train/{k}'] = v.mean()
        # if self.trainer.is_global_zero:
        #     wandb.log(wandb_log)

        self.log('train/loss', loss, batch_size=x[0].shape[0], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """validation step on self.device, called automaticly by PytorchLightning"""
        x, cls_label, sed_label, doa_label, num_spk, paras = batch  # x: [B,C,T], ys: [B,Spk,C,T]
        scaling_factor = torch.amax(torch.abs(x), dim=(1, 2), keepdim=True)
        x = x/(scaling_factor+1e-8)  # normalize the input
        B, S = cls_label.shape

        y_hat, y_ane_hat, y_rev_hat, cls_out, sed_out, doa_out, onoff_out = self.forward(x)
        # float32 loss calculation
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            loss, perms, logit_cls_out, loss_log = loss_fn(self, cls_out, sed_out, doa_out, onoff_out, cls_label, sed_label, doa_label, mode='test', weight=self.loss_weight)
            batch_idx = torch.arange(B, device=loss.device).unsqueeze(1).expand(B, S)
            # y_hat = y_hat[batch_idx, :, perms, ...]
            cls_out = cls_out[batch_idx, perms, ...]
            sed_out = sed_out[batch_idx, perms, ...]
            doa_out = doa_out[batch_idx, perms, ...]
            onoff_out = onoff_out[batch_idx, perms, ...]
            # yr_hat = y_hat[:, 0] # [B, S, T]

        # append results
        self.cls_out.append(torch.argmax(cls_out, dim=-1).float())
        self.cls_label.append(cls_label)
        self.features.append(cls_out)
        # self.onoff_out.append((torch.sigmoid(onoff_out) > 0.5).float() * (doa_out.norm(dim=-1) > 0.5).float())
        self.onoff_out.append((torch.sigmoid(onoff_out) > 0.5).float())
        self.sed_label.append(sed_label)
        self.doa_out.append(doa_out)
        self.doa_label.append(doa_label)

    def on_validation_epoch_end(self) -> None:
        """calculate heavy metrics for every N epochs"""
        # if not self.trainer.sanity_checking:
        #     self.val_wandb_log.update({'learning rate': self.optimizers().param_groups[0]['lr']})
        GS.on_validation_epoch_end(self=self, cpu_metric_input=self.val_cpu_metric_input, N=1)

    def on_test_epoch_start(self):
        self.exp_save_path = self.trainer.logger.log_dir
        os.makedirs(self.exp_save_path, exist_ok=True)
        self.results, self.cpu_metric_input = [], []

    def on_test_epoch_end(self):
        results_sample = GS.on_test_epoch_end(self=self, results=self.results, cpu_metric_input=self.cpu_metric_input, exp_save_path=self.exp_save_path)
        # SELD metrics
        if self.trainer.is_global_zero:
            data_for_seld_metrics = {'er': [], 'f1': [], 'le': [], 'lr': [], 'seld': []}

            feature_out_global = []
            cls_out_global = []
            cls_label_global = []
            onoff_out_global = []
            sed_out_global = []
            sed_label_global = []
            doa_outs_global = []
            doa_labels_global = []

            for wavname, data in results_sample.items():
                assert wavname not in data_for_seld_metrics.keys(), f"Duplicate wavname {wavname}"
                num_segment = len(data['features'])
                sed_outs = []
                for seg in range(num_segment):
                    active_mask = (torch.sigmoid(data['onoff_out'][seg]) > 0.5).float()
                    cls_binary = torch.zeros_like(data['features'][seg]).scatter_(-1, torch.argmax(data['features'][seg], dim=-1, keepdim=True), 1)
                    sed_out = active_mask.unsqueeze(-1) * cls_binary.unsqueeze(2)
                    sed_outs.append(sed_out)

                    feature_out_global.append(data['features'][seg])
                    cls_out_global.append(data['cls_out'][seg])
                    cls_label_global.append(data['cls_label'][seg])
                    onoff_out_global.append(data['onoff_out'][seg])
                    sed_out_global.append(data['sed_out'][seg])
                    sed_label_global.append(data['sed_label'][seg])
                    doa_outs_global.append(data['doa_out'][seg])
                    doa_labels_global.append(data['doa_label'][seg])

            active_mask = torch.cat(onoff_out_global, dim=0).float()
            cls_binary = torch.zeros_like(torch.cat(feature_out_global, dim=0)).scatter_(-1, torch.argmax(torch.cat(feature_out_global, dim=0), dim=-1, keepdim=True), 1)
            sed_out = active_mask.unsqueeze(-1) * cls_binary.unsqueeze(2)
            sed_label = torch.cat(sed_label_global, dim=0)
            doa_out = torch.cat(doa_outs_global, dim=0)
            doa_label = torch.cat(doa_labels_global, dim=0)
            # er, f1, le, lr, sc = SELD_metrics(sed_out, sed_label, doa_out, doa_label)
            # print(f"ER={er:.2f}, F1={f1:.2f}, LE={le:.2f}, LR={lr:.2f}, SELD={sc:.3f}")
            er, f1, le, lr, sc, f1_c, le_c, lr_c = SELD_metrics_c(sed_out, sed_label, doa_out, doa_label, per_class=True)
            print(f"ER={er:.2f}, F1={f1:.2f}, LE={le:.2f}, LR={lr:.2f}, SELD={sc:.3f}")
            class_names = ["Female speech", "Male speech", "Clapping", "Telephone", "Laughter", "Domestic sound", "Walk and footsteps", "Door", "Music", "Musical instrument", "Water tap", "Bell", "Knock"]
            for i, (f1_, le_, lr_) in enumerate(zip(f1_c, le_c, lr_c)):
                class_name = class_names[i] if i < len(class_names) else f"Class {i}"
                print(f"{class_name:<24} | F1: {f1_:.2f} | LE: {le_:.2f} | LR: {lr_:.2f}")

            # # 1) Save histograms along ER or LE
            # save_er_histogram(torch.tensor(self.er_values), self.exp_save_path + '/er_histogram.png')
            # save_le_histogram(torch.tensor(self.le_values), self.exp_save_path + '/le_histogram.png')
            # # # 2) Save confusion matrix
            cls_pred = torch.cat(cls_out_global, dim=1).cpu().numpy()
            cls_true = torch.cat(cls_label_global, dim=1).cpu().numpy()
            save_confusion_matrix(cls_true, cls_pred, self.exp_save_path + '/confusion_matrix.png')
            # # # 3) Save TSNE plot (Class)
            features = torch.cat(feature_out_global, dim=1).to(torch.float32).cpu().numpy()
            labels = torch.cat(cls_label_global, dim=1).cpu().numpy()
            plot_tsne(features, labels, self.exp_save_path + '/T-SNE_class', selected_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], perplexity=30, random_state=42)
            # # # 5) PCA analysis (Class)
            # f_features = torch.cat(self.f_features, dim=0).to(torch.float32).cpu().numpy()
            # plot_pca_from_feature_tensor(f_features, save_path=self.exp_save_path + '/pca_result.png')
            # plot_pca_per_class(f_features, labels, save_path=self.exp_save_path + '/pca_result_c.png')

    def test_step(self, batch, batch_idx):
        x, cls_label, sed_label, doa_label, num_spk, paras = batch  # x: [B,C,T], ys: [B,Spk,C,T]
        # x = x/(torch.max(torch.abs(x), dim=-1).values.unsqueeze(-1)+1e-8)  # normalize the input
        scaling_factor = torch.amax(torch.abs(x), dim=(1, 2), keepdim=True)
        x = x/(scaling_factor+1e-8)  # normalize the input
        B, S = cls_label.shape
        # num_spk = (num_spk > -1).sum().item()

        sample_rate = 16000 if 'sample_rate' not in paras[0] else paras[0]['sample_rate']
        y_hat, y_ane_hat, y_rev_hat, cls_out, sed_out, doa_out, onoff_out = self.forward(x)
        # float32 loss calculation
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            loss, perms, _, loss_log = loss_fn(self, cls_out, sed_out, doa_out, onoff_out, cls_label, sed_label, doa_label, mode='test', weight=None)
            batch_idx = torch.arange(B, device=loss.device).unsqueeze(1).expand(B, S)
            y_hat = y_hat[batch_idx, :, perms, ...]
            cls_out = cls_out[batch_idx, perms, ...]
            sed_out = sed_out[batch_idx, perms, ...]
            doa_out = doa_out[batch_idx, perms, ...]
            onoff_out = onoff_out[batch_idx, perms, ...]
            yr_hat = y_hat[:, :, 0] # [B, S, T]

        # # write results & infos
        wavname = '_'.join(paras[0]['wavname'].split('_')[:-1])
        result_dict = {'id': paras[0]['index'], 'wavname': wavname}
        for k, v in loss_log.items():
            if v is not None:
                result_dict[k] = v.mean().item()

        # calculate metrics, input_metrics, improve_metrics on GPU
        metrics, input_metrics, imp_metrics = cal_metrics_functional(self.metrics, yr_hat[0, :num_spk[0]], yr_hat[0, :num_spk[0]], x[:, self.ref_channel].expand_as(yr_hat[0, :num_spk[0]]), sample_rate, device_only='gpu')
        result_dict.update(input_metrics)
        result_dict.update(imp_metrics)
        result_dict.update(metrics)
        result_dict.update({'features': cls_out.float()})
        result_dict.update({'cls_out': torch.argmax(cls_out, dim=-1).float()})
        result_dict.update({'cls_label': cls_label.float()})
        result_dict.update({'sed_out': sed_out.float()})
        result_dict.update({'sed_label': sed_label.float()})
        result_dict.update({'doa_out': doa_out.float()})
        result_dict.update({'doa_label': doa_label.float()})
        # if self.onoff_out == None:
        #     result_dict.update({'onoff_out': (torch.norm(doa_out, dim=-1) > 0.5).float()})
        # else:
        #     result_dict.update({'onoff_out': ((torch.norm(doa_out, dim=-1) > 0.5) * (torch.sigmoid(onoff_out) > 0.5)).float()})
        result_dict.update({'onoff_out': (torch.sigmoid(onoff_out) > 0.5).float()})
        # self.f_features.append(f_feat)

        # Store input metrics for CPU usage
        self.cpu_metric_input.append((self.metrics, yr_hat[0, :num_spk[0]].detach().cpu(), yr_hat[0, :num_spk[0]].detach().cpu(), x[:, self.ref_channel].expand_as(yr_hat[0, :num_spk[0]]).detach().cpu(), sample_rate, 'cpu'))
        # Write examples if needed
        # if self.write_examples < 0 or paras[0]['index'] < self.write_examples:
        # if any(cls_label[0] == 7) or any(cls_label[0] == 12):
        #     # Plot graph of result of CASA (USS, SED, DoAE)
        #     GS.test_setp_write_example(
        #         self=self,
        #         xr=x[:, self.ref_channel],
        #         yr=None,
        #         yr_hat=yr_hat,
        #         sample_rate=sample_rate,
        #         paras=paras,
        #         result_dict=result_dict,
        #         wavname=wavname,
        #         exp_save_path=self.exp_save_path,
        #     )
        #     index = paras[0]['index']
        #     # X = X.squeeze(0)
        #     # mu = mu.squeeze(-1).to(torch.float32)
        #     # sigma = sigma.squeeze(-1).to(torch.float32)
        #     # sio.savemat(f"{self.exp_save_path}/examples/{index}/X.mat", {'X': X.cpu().numpy()})
        #     # [pd.DataFrame(tensor.cpu().numpy().round(5)).to_csv(f"{self.exp_save_path}/examples/{index}/{name}.csv", index=False, header=False) for tensor, name in zip([mu, sigma], ["mu", "sigma"])]
        #     save_path = os.path.join(self.exp_save_path, f'examples/{index}')
        #     # visualize_sed(onoff_out.float(), pp_onoff_out.float(), doa_out, doa_label, delta, max_idx, min_idx, cls_out, cls_label, save_path)
        #     visualize_out(yr_hat, sed_out, doa_out, sed_label, doa_label, ((torch.norm(doa_out, dim=-1) > 0.5) * (torch.sigmoid(onoff_out) > 0.5)).unsqueeze(-1).float(), save_path)
        
        if 'metrics' in paras[0]:
            del paras[0]['metrics']  # remove circular reference
        result_dict['paras'] = paras[0]
        self.results.append(result_dict)
        return result_dict

    def predict_step(self, batch, batch_idx):
        """predict step on self.device, could be called dirctly or by PytorchLightning automatically using predict dataset
        Args:
            batch: x or (x, ys, paras). shape of x [B, C, T]

        Returns:
            Tensor: ys_hat, shape [B, Spk, T]
        """
        x = batch
        yr = None

        # forward & loss
        yr_hat, cls_out = self.forward(x)

        if self.sep_loss.is_scale_invariant_loss:
            x_ref = x[:, self.ref_channel, :]
            yr_hat = recover_scale(preds=yr_hat, mixture=x_ref, scale_src_together=True if self.sep_loss.loss_func == neg_sa_sdr else False, norm_if_exceed_1=False)

        if yr is not None:  # reorder yr_hat if given yr
            _, perms = pit(preds=yr_hat, target=yr, metric_func=si_sdr, eval_func='min')
            yr_hat = pit_permutate(preds=yr_hat, perm=perms)

        # normalize the audios so that the maximum doesn't exceed 1
        if self.norm_if_exceed_1:
            max_vals = torch.max(torch.abs(yr_hat), dim=-1).values
            norm = torch.where(max_vals > 1, max_vals, 1)
            yr_hat = yr_hat / norm.unsqueeze(-1)

        return yr_hat

    def on_predict_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        GS.on_predict_batch_end(self=self, outputs=outputs, batch=batch)

    def configure_optimizers(self):
        """configure optimizer and lr_scheduler"""
        return GS.configure_optimizers(
            self=self,
            optimizer=self.optimizer[0],
            optimizer_kwargs=self.optimizer[1],
            monitor='val/metric',
            lr_scheduler=self.lr_scheduler[0] if self.lr_scheduler is not None else None,
            lr_scheduler_kwargs=self.lr_scheduler[1] if self.lr_scheduler is not None else None,
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        GS.on_load_checkpoint(self=self, checkpoint=checkpoint, ensemble_opts=self.ensemble, compile=self.compile)


class TrainCLI(BaseCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # # EarlyStopping
        # parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        # early_stopping_defaults = {
        #     "early_stopping.monitor": "val/metric",
        #     "early_stopping.patience": 30,
        #     "early_stopping.mode": "max",
        #     "early_stopping.min_delta": 0.1,
        # }
        # parser.set_defaults(early_stopping_defaults)

        # ModelCheckpoint
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_metric{val/metric:.4f}",
            "model_checkpoint.monitor": "val/metric",
            "model_checkpoint.mode": "min",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": 3,  # save all checkpoints
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True
        }
        parser.set_defaults(model_checkpoint_defaults)

        self.add_model_invariant_arguments_to_parser(parser)



if __name__ == '__main__':
    # python SharedTrainer.py --help
    cli = TrainCLI(
        TrainModule,
        pl.LightningDataModule,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        subclass_mode_data=True,
    )