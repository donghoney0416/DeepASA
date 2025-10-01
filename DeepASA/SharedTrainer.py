from models.utils.base_cli import BaseCLI
# import BaseCLI at the beginning

import os
from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
from jsonargparse import lazy_instance
from packaging.version import Version
from torch import Tensor
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from torchmetrics.functional.audio import signal_noise_ratio as snr
from sklearn.metrics import accuracy_score, f1_score
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.functional.audio import speech_reverberation_modulation_energy_ratio
import models.utils.general_steps_orig as GS
from models.io.loss import *
from models.io.norm import Norm
from models.io.stft import STFT, GaborSTFT
from models.utils.metrics import (cal_metrics_functional, recover_scale)
from models.utils.base_cli import BaseCLI
from models.utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback
from models.utils.SELD_evaluation_metrics import *
from visualize_ASA import *
from visualize_SELD import save_confusion_matrix, plot_onoff_histogram, plot_tsne, visualize_att_map


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
        stft: GaborSTFT = GaborSTFT(n_fft=640, n_hop=320, win_len=640, num_frames=201),
        norm: Norm = Norm(mode='utterance'),
        loss: Loss = Loss(loss_func=neg_sa_sdr, pit=True),
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
        self.sep_loss = Loss(loss_func=neg_sa_sdr, pit=True)
        self.cls_loss = AudioClassificationLoss()
        self.sed_loss = nn.MSELoss()
        self.doa_loss = nn.MSELoss()
        self.onoff_loss = nn.BCEWithLogitsLoss()

        # Analysis tool
        self.cls_out = []
        self.cls_label = []
        self.sdr_data = {class_index: {'input': [], 'output': []} for class_index in range(13)}
        self.sdr_cls_data = {class_index: {'imp': [], 'pred': []} for class_index in range(13)}
        self.n_spk_data = {'n_spk': [], 'si_sdri': []}
        self.num_spk_hat = []
        self.num_spk = []
        self.features = []
        self.onoff_out = []
        self.onoff_label = []
        self.le = []

        self.val_cpu_metric_input = []
        self.norm_if_exceed_1 = True
        self.name = type(arch).__name__

        # save other parameters to `self`
        for k, v in args.items():
            if k == 'self' or k == '__class__' or hasattr(self, k):
                continue
            setattr(self, k, v)

    def on_train_start(self):
        """Called by PytorchLightning automatically at the start of training"""
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

        sep_out, ane_out, rev_out, noise_out, cls_out, sed_out, doa_out, onoff_out = self.arch(X)  # [B,CM,F,T], complex
        # MIMO separation
        if sep_out != None:
            yr_hat = self.process_output(sep_out, B, C, F, T, x.shape, self.norm, self.stft, norm_paras, stft_paras, istft)
            yr_ane_hat, yr_rev_hat = None, None
        elif ane_out != None:
            yr_ane_hat = self.process_output(ane_out, B, C, F, T, x.shape, self.norm, self.stft, norm_paras, stft_paras, istft)
            yr_rev_hat = self.process_output(rev_out, B, C, F, T, x.shape, self.norm, self.stft, norm_paras, stft_paras, istft)
            yr_hat = None
        if noise_out != None:
            yn_hat = self.process_output(noise_out, B, C, F, T, x.shape, self.norm, self.stft, norm_paras, stft_paras, istft).squeeze(2)
        else:
            yn_hat = None

        return yr_hat, yr_ane_hat, yr_rev_hat, yn_hat, cls_out, sed_out, doa_out, onoff_out

    def training_step(self, batch, batch_idx):
        """training step on self.device, called automaticly by PytorchLightning"""
        x, y_ane, y_rev, num_spk, cls_label, sed_label, doa_label, paras = batch  # x: [B,C,T], ys: [B,Spk,C,T]
        y = y_ane + y_rev
        yr = y[:, :, self.ref_channel, :]
        yr_ane = y_ane[:, :, self.ref_channel, :]
        yr_rev = y_rev[:, :, self.ref_channel, :]
        yn = (x - y.sum(dim=1))  # [B,C,T]

        y_hat, y_ane_hat, y_rev_hat, yn_hat, cls_out, sed_out, doa_out, onoff_out = self.forward(x)
        y_hat = y_ane_hat + y_rev_hat if y_hat == None else y_hat
        # float32 loss calculation
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            # USS loss
            # MIMO separation
            sep_loss_nref = sum([self.sep_loss(preds=y_hat[:, i], target=y[:, :, i], reorder=True, reduce_batch=True)[0] for i in range(0, y.shape[2]-1)])
            sep_loss, perms, yr_hat = self.sep_loss(preds=y_hat[:, 0], target=yr, reorder=True, reduce_batch=True)
            if y_ane_hat != None:
                # Anechoic source separation
                sep_loss_nref += sum([self.sep_loss(preds=y_ane_hat[:, i], target=y_ane[:, :, i], reorder=True, reduce_batch=True)[0] for i in range(0, y_ane.shape[2]-1)])
                sep_loss_ane = self.sep_loss(preds=y_ane_hat[:, 0], target=yr_ane, reorder=True, reduce_batch=True)[0]
                # Reverberant source separation
                sep_loss_nref += sum([self.sep_loss(preds=y_rev_hat[:, i], target=y_rev[:, :, i], reorder=True, reduce_batch=True)[0] for i in range(0, y_rev.shape[2]-1)])
                sep_loss_rev = self.sep_loss(preds=y_rev_hat[:, 0], target=yr_rev, reorder=True, reduce_batch=True)[0]
                sep_loss = sep_loss + sep_loss_ane + sep_loss_rev
            if yn_hat != None:
                sep_loss -= 0.01*si_sdr(yn_hat, yn).mean()  # add noise loss
            # SED loss
            cls_loss = 0.
            if cls_out != None:
                cls_out = pit_permutate(preds=cls_out, perm=perms)
                cls_loss += sum([self.cls_loss(cls_out[:, i], cls_label[:, i]) for i in range(y.shape[1])])/y.shape[1]
            if sed_out != None:
                sed_out = pit_permutate(preds=sed_out, perm=perms)
                cls_loss += self.sed_loss(sed_out, sed_label)
            # DoAE loss
            doa_loss = 0.
            if doa_out != None:
                doa_out = pit_permutate(preds=doa_out, perm=perms)
                doa_loss = self.doa_loss(doa_out, doa_label)
            # On/Offset loss
            onoff_loss = 0.
            if onoff_out != None:
                onoff = pit_permutate(preds=onoff_out, perm=perms)
                onoff_gt = (torch.norm(doa_label, dim=-1) > 0.5).float()
                onoff_loss = self.onoff_loss(onoff.reshape(-1), onoff_gt.reshape(-1))
            # Multi-task loss
            loss = sep_loss + 0.1*sep_loss_nref + cls_loss + doa_loss + onoff_loss

        self.log('train/loss', loss, batch_size=x[0].shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """validation step on self.device, called automaticly by PytorchLightning"""
        x, y_ane, y_rev, num_spk, cls_label, sed_label, doa_label, paras = batch  # x: [B,C,T], ys: [B,Spk,C,T]
        y = y_ane + y_rev
        yr = y[:, :, self.ref_channel, :]
        yr_ane = y_ane[:, :, self.ref_channel, :]
        yr_rev = y_rev[:, :, self.ref_channel, :]
        yn = (x - y.sum(dim=1))  # [B,C,T]

        y_hat, y_ane_hat, y_rev_hat, yn_hat, cls_out, sed_out, doa_out, onoff_out = self.forward(x)
        y_hat = y_ane_hat + y_rev_hat if y_hat == None else y_hat
        # float32 loss calculation
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            # USS loss
            # MIMO separation
            sep_loss_nref = sum([self.sep_loss(preds=y_hat[:, i], target=y[:, :, i], reorder=True, reduce_batch=True)[0] for i in range(0, y.shape[2]-1)])
            sep_loss, perms, yr_hat = self.sep_loss(preds=y_hat[:, 0], target=yr, reorder=True, reduce_batch=True)
            if y_ane_hat != None:
                # Anechoic source separation
                sep_loss_nref += sum([self.sep_loss(preds=y_ane_hat[:, i], target=y_ane[:, :, i], reorder=True, reduce_batch=True)[0] for i in range(0, y_ane.shape[2]-1)])
                sep_loss_ane, _, yr_ane_hat = self.sep_loss(preds=y_ane_hat[:, 0], target=yr_ane, reorder=True, reduce_batch=True)
                # Reverberant source separation
                sep_loss_nref += sum([self.sep_loss(preds=y_rev_hat[:, i], target=y_rev[:, :, i], reorder=True, reduce_batch=True)[0] for i in range(0, y_rev.shape[2]-1)])
                sep_loss_rev, _, yr_rev_hat = self.sep_loss(preds=y_rev_hat[:, 0], target=yr_rev, reorder=True, reduce_batch=True)
                sep_loss = sep_loss + sep_loss_ane + sep_loss_rev
            if yn_hat != None:
                sep_loss -= 0.01*si_sdr(yn_hat, yn).mean()  # add noise loss
            # SED loss
            cls_loss = 0.
            if cls_out != None:
                cls_out = pit_permutate(preds=cls_out, perm=perms)
                cls_loss += sum([self.cls_loss(cls_out[:, i], cls_label[:, i]) for i in range(y.shape[1])])/y.shape[1]
            if sed_out != None:
                sed_out = pit_permutate(preds=sed_out, perm=perms)
                cls_loss += self.sed_loss(sed_out, sed_label)
            # DoAE loss
            doa_loss = 0.
            if doa_out != None:
                doa_out = pit_permutate(preds=doa_out, perm=perms)
                doa_loss = self.doa_loss(doa_out, doa_label)
            # On/Offset loss
            onoff_loss = 0.
            if onoff_out != None:
                onoff_out = pit_permutate(preds=onoff_out, perm=perms)
                onoff_gt = (torch.norm(doa_label, dim=-1) > 0.5).float()
                onoff_loss = self.onoff_loss(onoff_out.reshape(-1), onoff_gt.reshape(-1))
            # Multi-task loss
            loss = sep_loss + 0.1*sep_loss_nref + cls_loss + doa_loss + onoff_loss

        # USS evaluation
        si_sdr_val, sdr_val, si_sdr_ane, si_sdr_rev = 0., 0., 0., 0.
        for batch_idx in range(x.shape[0]):
            if y_hat != None:
                si_sdr_val += si_snr(yr_hat[batch_idx, :num_spk[batch_idx]], yr[batch_idx, :num_spk[batch_idx]]).mean()
                sdr_val += snr(yr_hat[batch_idx, :num_spk[batch_idx]], yr[batch_idx, :num_spk[batch_idx]]).mean()
                # Dereverberation evaluation
                si_sdr_ane += si_snr(yr_ane_hat[batch_idx, :num_spk[batch_idx]], yr_ane[batch_idx, :num_spk[batch_idx]]).mean()
                si_sdr_rev += si_snr(yr_rev_hat[batch_idx, :num_spk[batch_idx]], yr_rev[batch_idx, :num_spk[batch_idx]]).mean()
        si_sdr_val /= x.shape[0]
        sdr_val /= x.shape[0]
        si_sdr_ane /= x.shape[0]
        si_sdr_rev /= x.shape[0]
        # SELD evaluation
        active_mask = ((torch.norm(doa_out, dim=-1) > 0.5) * (torch.sigmoid(onoff_out) > 0.5)).float()
        cls_binary = torch.zeros_like(cls_out).scatter_(-1, torch.argmax(cls_out, dim=-1, keepdim=True), 1)
        sed_out = active_mask.unsqueeze(-1) * cls_binary.unsqueeze(2)                     # [B, S, T, C]
        error_rate, f1, le, lr, sc = SELD_metrics(sed_out, sed_label, doa_out, doa_label)

        # loss function
        self.log('val/loss', loss, sync_dist=True, batch_size=x.shape[0])
        # USS evaluation
        self.log('si_sdr', si_sdr_val, sync_dist=True, batch_size=x.shape[0])
        self.log('sdr', sdr_val, sync_dist=True, batch_size=x.shape[0])
        self.log('si_sdr_ane', si_sdr_ane, sync_dist=True, batch_size=x.shape[0])
        self.log('si_sdr_rev', si_sdr_rev, sync_dist=True, batch_size=x.shape[0])
        # # classification evaluation
        self.log('ER', error_rate, sync_dist=True, batch_size=x.shape[0])
        self.log('F1', f1, sync_dist=True, batch_size=x.shape[0])
        # # DoA estimation evaluation
        self.log('LE', le, sync_dist=True, batch_size=x.shape[0])
        self.log('LR', lr, sync_dist=True, batch_size=x.shape[0])
        # # logging
        val_metric = {'loss': -loss, 'SI-SDR': si_sdr_val, 'SDR': sdr_val, 'SI-SDR_ane': si_sdr_ane, 'SI-SDR_rev': si_sdr_rev, 'ER': error_rate, 'F1': f1, 'LE': le, 'LR': lr, 'SC': sc}[self.val_metric]
        self.log('val/metric', val_metric, sync_dist=True, batch_size=x.shape[0])  # log val/metric for checkpoint picking

    def on_validation_epoch_end(self) -> None:
        """calculate heavy metrics for every N epochs"""
        GS.on_validation_epoch_end(self=self, cpu_metric_input=self.val_cpu_metric_input, N=5)
        # logging for every epoch
        self.log_file = os.path.join(self.trainer.logger.log_dir, "result.txt")
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        if not os.path.exists(self.log_file) and torch.distributed.get_rank() == 0:
            with open(self.log_file, "w") as f:
                f.write(f"{'Epoch':<6}{'Loss':<10}{'SI-SDR':<10}{'SDR':<9}{'ER':<11}{'F1':<10}{'LE':<10}{'LR':<10}\n")
        epoch = self.trainer.current_epoch
        # Load recent metrics
        loss = round(self.trainer.callback_metrics.get("val/loss", torch.tensor(0.0)).item(), 2)
        si_sdr = round(self.trainer.callback_metrics.get("si_sdr", torch.tensor(0.0)).item(), 2)
        sdr = round(self.trainer.callback_metrics.get("sdr", torch.tensor(0.0)).item(), 2)
        er = round(self.trainer.callback_metrics.get("ER", torch.tensor(0.0)).item(), 2)
        f1 = round(self.trainer.callback_metrics.get("F1", torch.tensor(0.0)).item(), 2)
        le = round(self.trainer.callback_metrics.get("LE", torch.tensor(0.0)).item(), 2)
        lr = round(self.trainer.callback_metrics.get("LR", torch.tensor(0.0)).item(), 2)
        # save to log file
        with open(self.log_file, "a") as f:
            f.write(f"{epoch:<8}{loss:<10}{si_sdr:<10}{sdr:<10}{er:<10}{f1:<10}{le:<9}{lr:<10}\n")

    def on_test_epoch_start(self):
        self.exp_save_path = self.trainer.logger.log_dir
        os.makedirs(self.exp_save_path, exist_ok=True)
        self.results, self.cpu_metric_input = [], []

    def on_test_epoch_end(self):
        GS.on_test_epoch_end(self=self, results=self.results, cpu_metric_input=self.cpu_metric_input, exp_save_path=self.exp_save_path)
        plot_le_histogram(np.array(self.le), self.exp_save_path + '/le_histogram.png')
        # 1) Save confusion matrix
        in_bound = torch.max(torch.softmax(torch.cat(self.features, dim=1), dim=-1), dim=-1)[0] > 2 / 13
        cls_pred = torch.argmax(torch.cat(self.features, dim=1), dim=-1)
        cls_pred[~in_bound] = -1
        cls_pred = cls_pred.cpu().numpy()
        cls_true = torch.cat(self.cls_label, dim=1).cpu().numpy()
        save_ASA_confusion_matrix(cls_true, cls_pred, self.exp_save_path + '/confusion_matrix_ood.png')
        cls_pred = torch.cat(self.cls_out, dim=1).cpu().numpy()
        save_ASA_confusion_matrix(cls_true, cls_pred, self.exp_save_path + '/confusion_matrix.png')
        # 2) Save SI-SDR data along class
        scatter_plot(self.sdr_data, self.exp_save_path)
        box_plot(self.sdr_data, self.exp_save_path)
        # # 2-1) Save predicted class along SI-SDR improvement
        scatter_plot_cls(self.sdr_cls_data, self.exp_save_path)
        # # 3) Save SI-SDR data along n_spk
        violin_plot(self.n_spk_data, self.exp_save_path + '/violin_plot.png')
        save_confusion_matrix_nspk(self.num_spk_hat, self.num_spk, self.exp_save_path + '/confusion_matrix_nspk.png')
        # # 4) Onset/Offset analysis
        onoff_out = torch.cat(self.onoff_out, dim=1).squeeze(0).cpu().numpy()
        onoff_labels = torch.cat(self.onoff_label, dim=1).squeeze(0).cpu().numpy()
        plot_onoff_histogram(onoff_out, onoff_labels, self.exp_save_path + '/onoff_histogram.png')
        # # 5) Save T-SNE plot
        features = torch.cat(self.features, dim=1).to(torch.float32).cpu().numpy()
        labels = torch.cat(self.cls_label, dim=1).cpu().numpy()
        plot_tsne(features, labels, self.exp_save_path + '/T-SNE_class', selected_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], perplexity=30, random_state=42)

    def test_step(self, batch, batch_idx):
        x, y_ane, y_rev, num_spk, cls_label, sed_label, doa_label, paras = batch  # x: [B,C,T], ys: [B,Spk,C,T]
        # y = y_ane + y_rev
        yr = y[:, :, self.ref_channel, :]
        yr_ane = y_ane[:, :, self.ref_channel, :]
        yr_rev = y_rev[:, :, self.ref_channel, :]
        yn = (x - y.sum(dim=1))  # [B,C,T]

        y_hat, y_ane_hat, y_rev_hat, yn_hat, cls_out, sed_out, doa_out, onoff_out = self.forward(x)
        y_hat = y_ane_hat + y_rev_hat if y_hat == None else y_hat
        
        sample_rate = 16000 if 'sample_rate' not in paras[0] else paras[0]['sample_rate']
        # float32 loss calculation
        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            # use float 32 precision for validation and test
            autocast = torch.autocast(device_type=self.device.type, dtype=torch.float32)
            autocast.__enter__()
        # USS loss
        # MIMO separation
            sep_loss_nref = sum([self.sep_loss(preds=y_hat[:, i], target=y[:, :, i], reorder=True, reduce_batch=True)[0] for i in range(0, y.shape[2]-1)])
            sep_loss, perms, yr_hat = self.sep_loss(preds=y_hat[:, 0], target=yr, reorder=True, reduce_batch=True)
        if y_ane_hat != None:
            # Entire source separation
            sep_loss_nref = sum([self.sep_loss(preds=y_hat[:, i], target=y[:, :, i], reorder=True, reduce_batch=True)[0] for i in range(0, y.shape[2]-1)])
            sep_loss, perms, yr_hat = self.sep_loss(preds=y_hat[:, 0], target=yr, reorder=True, reduce_batch=True)
            # Anechoic source separation
            sep_loss_nref += sum([self.sep_loss(preds=y_ane_hat[:, i], target=y_ane[:, :, i], reorder=True, reduce_batch=True)[0] for i in range(0, y_ane.shape[2]-1)])
            sep_loss += self.sep_loss(preds=y_ane_hat[:, 0], target=yr_ane, reorder=True, reduce_batch=True)[0]
            yr_ane_hat = self.sep_loss(preds=y_ane_hat[:, 0], target=yr_ane, reorder=True, reduce_batch=True)[2]
            # Reverberant source separation
            sep_loss_nref += sum([self.sep_loss(preds=y_rev_hat[:, i], target=y_rev[:, :, i], reorder=True, reduce_batch=True)[0] for i in range(0, y_rev.shape[2]-1)])
            sep_loss += self.sep_loss(preds=y_rev_hat[:, 0], target=yr_rev, reorder=True, reduce_batch=True)[0]
            yr_rev_hat = self.sep_loss(preds=y_rev_hat[:, 0], target=yr_rev, reorder=True, reduce_batch=True)[2]
        if yn_hat != None:
            sep_loss -= 0.01*si_sdr(yn_hat, yn).mean()  # add noise loss
        # SED loss
        cls_loss = 0.
        if cls_out != None:
            cls_out = pit_permutate(preds=cls_out, perm=perms)
            cls_loss += sum([self.cls_loss(cls_out[:, i], cls_label[:, i]) for i in range(y.shape[1])])/y.shape[1]
        if sed_out != None:
            sed_out = pit_permutate(preds=sed_out, perm=perms)
            cls_loss += self.sed_loss(sed_out, sed_label)
        # DoAE loss
        doa_out = pit_permutate(preds=doa_out, perm=perms)
        doa_loss = self.doa_loss(doa_out, doa_label)
        # On/Offset loss
        onoff_loss = 0.
        if onoff_out != None:
            onoff_out = pit_permutate(preds=onoff_out, perm=perms)
            onoff_gt = (torch.norm(doa_label, dim=-1) > 0.5).float()
            onoff_loss = self.onoff_loss(onoff_out.reshape(-1), onoff_gt.reshape(-1))
        # Multi-task loss
        loss = sep_loss + 0.1*sep_loss_nref + cls_loss + doa_loss + onoff_loss

        # ## evaluation
        # # USS evaluation
        si_sdr_val, sdr_val, si_sdr_ane, si_sdr_rev = 0., 0., 0., 0.
        for batch_idx in range(x.shape[0]):
            if y_hat != None:
                si_sdr_val += si_snr(yr_hat[batch_idx, :num_spk[batch_idx]], yr[batch_idx, :num_spk[batch_idx]]).mean()
                sdr_val += snr(yr_hat[batch_idx, :num_spk[batch_idx]], yr[batch_idx, :num_spk[batch_idx]]).mean()
                # Dereverberation evaluation
                si_sdr_ane += si_snr(x[batch_idx, self.ref_channel].unsqueeze(0).repeat(num_spk[batch_idx], 1), yr_ane[batch_idx, :num_spk[batch_idx]]).mean()
                si_sdr_rev += si_snr(x[batch_idx, self.ref_channel].unsqueeze(0).repeat(num_spk[batch_idx], 1), yr_rev[batch_idx, :num_spk[batch_idx]]).mean()
                si_sdr_ane += si_snr(yr_ane_hat[batch_idx, :num_spk[batch_idx]], yr_ane[batch_idx, :num_spk[batch_idx]]).mean()
                si_sdr_rev += si_snr(yr_rev_hat[batch_idx, :num_spk[batch_idx]], yr_rev[batch_idx, :num_spk[batch_idx]]).mean()
        si_sdr_val /= x.shape[0]
        sdr_val /= x.shape[0]
        si_sdr_ane /= x.shape[0]
        si_sdr_rev /= x.shape[0]
        # # SELD evaluation
        active_mask = ((torch.norm(doa_out, dim=-1) > 0.5) * (torch.sigmoid(onoff_out) > 0.5)).float()
        cls_binary = torch.zeros_like(cls_out).scatter_(-1, torch.argmax(cls_out, dim=-1, keepdim=True), 1)
        sed_out = active_mask.unsqueeze(-1) * cls_binary.unsqueeze(2)                     # [B, S, T, C]
        error_rate, f1, le, lr, sc = SELD_metrics(sed_out, sed_label, doa_out, doa_label)

        # write results & infos
        wavname = os.path.basename(f"{paras[0]['index']}.wav")
        result_dict = {'id': batch_idx, 'wavname': wavname, self.sep_loss.name: loss.item()}
        # consider only n_spk speakers
        # yr_hat, yr = [x[:, :num_spk] for x in [yr_hat, yr]]

        # loss function
        self.log('val/loss', loss, sync_dist=True, batch_size=x.shape[0])
        # USS evaluation
        self.log('si_sdr', si_sdr_val, sync_dist=True, batch_size=x.shape[0])
        self.log('sdr', sdr_val, sync_dist=True, batch_size=x.shape[0])
        # Dereverberation evaluation
        self.log('si_sdr_ane', si_sdr_ane, sync_dist=True, batch_size=x.shape[0])
        self.log('si_sdr_rev', si_sdr_rev, sync_dist=True, batch_size=x.shape[0])
        # SED evaluation
        self.log('ER', error_rate, sync_dist=True, batch_size=x.shape[0])
        self.log('F1', f1, sync_dist=True, batch_size=x.shape[0])
        # DoA estimation evaluation
        self.log('LE', le, sync_dist=True, batch_size=x.shape[0])
        self.log('LR', lr, sync_dist=True, batch_size=x.shape[0])
        # SELD score
        self.log('SC', sc, sync_dist=True, batch_size=x.shape[0])
        # logging
        val_metric = {'loss': -loss, 'SI-SDR': si_sdr_val, 'SDR': sdr_val, 'ER': error_rate, 'F1': f1, 'LE': le, 'LR': lr, 'SC': sc}[self.val_metric]
        self.log('val/metric', val_metric, sync_dist=True, batch_size=x.shape[0])  # log val/metric for checkpoint picking

        # Recover scale if needed
        if self.sep_loss.is_scale_invariant_loss:
            x_ref = x[:, self.ref_channel, :]
            yr_hat = recover_scale(preds=yr_hat, mixture=x_ref, scale_src_together=True if self.sep_loss.loss_func == neg_sa_sdr else False, norm_if_exceed_1=False)

        x_ref = x[:, self.ref_channel, :]
        # Calculate metrics, input_metrics, imp_metrics on GPU
        metrics, input_metrics, imp_metrics = cal_metrics_functional(self.metrics, yr_hat[0, :num_spk[0]], yr[0, :num_spk[0]], x_ref.expand_as(yr[0, :num_spk[0]]), sample_rate, device_only='gpu')
        result_dict.update(input_metrics)
        result_dict.update(imp_metrics)
        result_dict.update(metrics)
        result_dict.update({
            'cls_label': cls_label.cpu().numpy(),
            'cls_out': cls_out.to(torch.float32).cpu().numpy(),
            'doa_label': doa_label.to(torch.float32).cpu().numpy().tolist(),
            'doa_out': doa_out.to(torch.float32).cpu().numpy().tolist(),
            'localization_error': le
        })
        # Store input metrics for CPU usage
        self.cpu_metric_input.append((self.metrics, yr_hat[0, :num_spk[0]].detach().cpu(), yr[0, :num_spk[0]].detach().cpu(), x_ref.expand_as(yr[0, :num_spk[0]]).detach().cpu(), sample_rate, 'cpu'))

        # Write examples if needed
        if self.write_examples < 0 or paras[0]['index'] < self.write_examples:
            # Plot graph of result of CASA (USS, SED, DoAE)
            GS.test_setp_write_example(
                self=self,
                xr=x[:, self.ref_channel],
                yr=yr[:, :num_spk[0]],
                yr_hat=yr_hat[:, :num_spk[0]],
                sample_rate=sample_rate,
                paras=paras,
                result_dict=result_dict,
                wavname=wavname,
                exp_save_path=self.exp_save_path,
            )
            save_path = os.path.join(self.exp_save_path, f'examples/{wavname[:-4]}/plot.png')
            onoff_out = torch.sigmoid(onoff_out) > 0.5
            visualize_out(yr_hat, sed_out, doa_out, yr, sed_label, doa_label, onoff_out, save_path)

        # 16-bit mixed precision handling if needed
        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            autocast.__exit__(None, None, None)

        # Final result dict update
        if 'metrics' in paras[0]:
            del paras[0]['metrics']  # Remove circular reference
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
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        early_stopping_defaults = {
            "early_stopping.monitor": "val/metric",
            "early_stopping.patience": 30,
            "early_stopping.mode": "max",
            "early_stopping.min_delta": 0.1,
        }
        parser.set_defaults(early_stopping_defaults)

        # ModelCheckpoint
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_metric{val/metric:.4f}",
            "model_checkpoint.monitor": "val/metric",
            "model_checkpoint.mode": "max",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": 5,  # save all checkpoints
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