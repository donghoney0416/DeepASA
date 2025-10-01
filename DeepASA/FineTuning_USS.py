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
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from torchmetrics.functional.audio import signal_noise_ratio as snr
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import models.utils.general_steps_ as GS
from models.io.loss import *
from models.io.norm import Norm
from models.io.stft import STFT, GaborSTFT
from models.utils.metrics import (cal_metrics_functional, recover_scale)
from models.utils.base_cli import BaseCLI
from models.utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback
import data_loaders
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd


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
        optimizer: Tuple[str, Dict[str, Any]] = ("Adam", {
            "lr": 0.0004
        }),
        lr_scheduler: Optional[Tuple[str, Dict[str, Any]]] = ('ReduceLROnPlateau', {
            'mode': 'min',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-4
        }),
        metrics: List[str] = ['SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ', 'eSTOI'],
        val_metric: str = 'loss',
        write_examples: int = 50,
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
        self.loss = loss
        self.n_spk_data = {'n_spk': [], 'si_sdri': [], 'sdri': []}

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

    def forward(self, x: Tensor, istft: bool = True) -> Tensor:
        """
        Args:
            x: [B,C,T]

        Returns:
            Tuple[Tensor, Tensor]: ys_hat
        """
        # obtain STFT X
        X, stft_paras = self.stft.stft(x[:, self.channels])  # [B,C,F,T], complex
        B, C, F, T = X.shape
        X, norm_paras = self.norm.norm(X, ref_channel=self.channels.index(self.ref_channel))
        X = X.permute(0, 2, 3, 1)  # B,F,T,C; complex
        X = torch.view_as_real(X).reshape(B, F, T, -1)  # B,F,T,2C

        # network process
        out, noise = self.arch(X)
        if not torch.is_complex(out):
            out = torch.view_as_complex(out.float().reshape(B, F, T, -1, 2))  # [B,F,T,Spk]
            noise = torch.view_as_complex(noise.float().reshape(B, F, T, -1, 2))  # [B,F,T,Spk]
        out = out.permute(0, 3, 1, 2)  # [B,Spk,F,T]
        noise = noise.permute(0, 3, 1, 2)  # [B,Spk,F,T]

        # to time domain
        Yr_hat = self.norm.inorm(out, norm_paras)
        Yn_hat = self.norm.inorm(noise, norm_paras)
        yr_hat = self.stft.istft(Yr_hat, stft_paras) if istft else torch.view_as_real(Yr_hat)
        yn_hat = self.stft.istft(Yn_hat, stft_paras) if istft else torch.view_as_real(Yn_hat)
        return yr_hat, yn_hat

    def training_step(self, batch, batch_idx):
        """training step on self.device, called automaticly by PytorchLightning"""
        x, ys, n_spk, paras = batch  # x: [B,C,T], ys: [B,Spk,C,T]
        yr = ys[:, :, self.ref_channel, :]
        yn = (x - ys.sum(dim=1))[:, self.ref_channel, :]  # [B,T]

        yr_hat, yn_hat = self.forward(x)

        # float32 loss calculation
        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                loss, perms, yr_hat = self.loss(preds=yr_hat, target=yr, reorder=False, reduce_batch=True)
                loss -= 0.01*si_sdr(yn_hat, yn).mean()  # add noise loss
        else:
            loss, perms, yr_hat = self.loss(preds=yr_hat, target=yr, reorder=False, reduce_batch=True)
            loss -= 0.01*si_sdr(yn_hat.squeeze(1), yn).mean()  # add noise loss

        self.log('train/' + self.loss.name, loss, batch_size=ys[0].shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """validation step on self.device, called automaticly by PytorchLightning"""
        x, ys, n_spk, paras = batch
        yr = ys[:, :, self.ref_channel, :]
        yn = (x - ys.sum(dim=1))[:, self.ref_channel, :]  # [B,T]

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            # use float 32 precision for validation and test
            autocast = torch.autocast(device_type=self.device.type, dtype=torch.float32)
            autocast.__enter__()

        # forward & loss
        yr_hat, yn_hat = self.forward(x)
        loss, perms, yr_hat = self.loss(preds=yr_hat, target=yr, reorder=True, reduce_batch=True)
        loss -= 0.01*si_sdr(yn_hat.squeeze(1), yn).mean()  # add noise loss
        # metrics
        sdr_val = 0
        si_sdr_val = 0
        for idx in range(yr_hat.shape[0]):
            sdr_val += snr(yr_hat[idx, :n_spk[idx]], yr[idx, :n_spk[idx]]).mean()
            si_sdr_val += si_snr(yr_hat[idx, :n_spk[idx]], yr[idx, :n_spk[idx]]).mean()
        sdr_val /= yr_hat.shape[0]
        si_sdr_val /= yr_hat.shape[0]

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            autocast.__exit__(None, None, None)

        # logging
        self.log('val/' + self.loss.name, loss, sync_dist=True, batch_size=ys.shape[0])
        val_metric = {'loss': -loss, 'si_sdr': si_sdr_val}[self.val_metric]
        self.log('val/metric', val_metric, sync_dist=True, batch_size=ys.shape[0])  # log val/metric for checkpoint picking

        # always computes the sdr/sisdr for the comparison of different runs
        self.log('val/sdr', sdr_val, sync_dist=True, batch_size=ys.shape[0])
        if self.loss.name != 'neg_si_sdr':
            # always computes the neg_si_sdr for the comparison of different runs in Tensorboard
            self.log('val/neg_si_sdr', -si_sdr_val, sync_dist=True, batch_size=ys.shape[0])

        # other heavy metrics: pesq
        # sample_rate = paras[0]['sample_rate']
        # yrs = [[
        #     ['nb_pesq'] if sample_rate == 8000 else ['nb_pesq', 'wb_pesq'],
        #     yr_hat.detach().cpu(),
        #     yr.detach().cpu(),
        #     None,
        #     sample_rate,
        #     'cpu',
        # ]]
        # self.val_cpu_metric_input += yrs

    def on_validation_epoch_end(self) -> None:
        """calculate heavy metrics for every N epochs"""
        GS.on_validation_epoch_end(self=self, cpu_metric_input=self.val_cpu_metric_input, N=5)

    def on_test_epoch_start(self):
        self.exp_save_path = self.trainer.logger.log_dir
        os.makedirs(self.exp_save_path, exist_ok=True)
        self.results, self.cpu_metric_input = [], []

    def violin_plot(self, nspk_data, path):
        df = pd.DataFrame(nspk_data)
        plt.figure(figsize=(6, 8))
        ax = sns.violinplot(x='n_spk', y='si_sdri', data=df, inner='point', palette='Set2')

        for i, n_spk_value in enumerate(sorted(df['n_spk'].unique())):  
            median_value = df[df['n_spk'] == n_spk_value]['si_sdri'].median()
            ax.scatter(i, median_value, color='black', s=100, marker='o', zorder=3)  
            ax.text(i, median_value + 1, f'{median_value:.2f}', color='black', ha='center', va='bottom', fontsize=10)

        plt.xlabel('J (Number of speakers)')
        plt.ylabel('SI-SDRi')
        plt.title('SI-SDRi distribution by number of speakers')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.ylim(-20, 60)

        plt.savefig(path, dpi=400)
        plt.close()

    def on_test_epoch_end(self):
        sdri = sum(self.n_spk_data["sdri"]) / len(self.n_spk_data["sdri"])
        si_sdri = sum(self.n_spk_data["si_sdri"]) / len(self.n_spk_data["si_sdri"])
        print(f'SUMMARY / SDRi:{sdri}, SI-SDRi: {si_sdri}')
        GS.on_test_epoch_end(self=self, results=self.results, cpu_metric_input=self.cpu_metric_input, exp_save_path=self.exp_save_path)
        self.violin_plot(self.n_spk_data, self.exp_save_path + '/violin_plot.png')

    def test_step(self, batch, batch_idx):
        x, ys, n_spk, paras = batch
        yr = ys[:, :, self.ref_channel, :]
        yn = (x - ys.sum(dim=1))[:, self.ref_channel, :]  # [B,T]
        x_ref = x[:, 0]
        sample_rate = 16000 if 'sample_rate' not in paras[0] else paras[0]['sample_rate']

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            # use float 32 precision for validation and test
            autocast = torch.autocast(device_type=self.device.type, dtype=torch.float32)
            autocast.__enter__()

        yr_hat, yn_hat = self.forward(x)
        loss, perms, yr_hat = self.loss(preds=yr_hat, target=yr, reorder=True, reduce_batch=True)
        loss -= 0.01*si_sdr(yn_hat.squeeze(1), yn).mean()  # add noise loss
        yr_hat = yr_hat[:, :n_spk[0]]
        yr = yr[:, :n_spk[0]]
        self.log('test/' + self.loss.name, loss, logger=False, batch_size=ys.shape[0])

        # write results & infos
        wavname = os.path.basename(f"{paras[0]['index']}.wav")
        result_dict = {'id': batch_idx, 'wavname': wavname, self.loss.name: loss.item()}

        # recover wav's original scale. solve min ||Y^T a - X||F to obtain the scales of the predictions of speakers, cuz sisdr will lose scale
        if self.loss.is_scale_invariant_loss:
            x_ref = x[:, self.ref_channel, :yr.shape[2]]
            yr_hat = recover_scale(preds=yr_hat, mixture=x_ref, scale_src_together=True if self.loss.loss_func == neg_sa_sdr else False, norm_if_exceed_1=False)

        for batch_idx in range(x.shape[0]):
            si_sdri = 0
            sdri = 0
            for source_idx in range(yr.shape[1]):
                if source_idx < n_spk[batch_idx]:
                    sdr_input = sdr(x[batch_idx, self.ref_channel], yr[batch_idx, source_idx]).item()
                    sdr_output = sdr(yr_hat[batch_idx, source_idx], yr[batch_idx, source_idx]).item()
                    sdri += sdr_output - sdr_input
                    si_sdr_input = si_sdr(x[batch_idx, self.ref_channel], yr[batch_idx, source_idx]).item()
                    si_sdr_output = si_sdr(yr_hat[batch_idx, source_idx], yr[batch_idx, source_idx]).item()
                    si_sdri += si_sdr_output - si_sdr_input
            si_sdri /= n_spk[batch_idx]
            sdri /= n_spk[batch_idx]
        # self.n_spk_data['n_spk'].append(n_spk.cpu().item())
        self.n_spk_data['si_sdri'].append(si_sdri)
        self.n_spk_data['sdri'].append(sdri)
        print(f'sdri: {sdri:.2f}, si_sdri: {si_sdri:.2f}')

        # calculate metrics, input_metrics, improve_metrics on GPU
        metrics, input_metrics, imp_metrics = cal_metrics_functional(self.metrics, yr_hat[0], yr[0], x_ref.expand_as(yr[0]), sample_rate, device_only='gpu')
        result_dict.update(input_metrics)
        result_dict.update(imp_metrics)
        result_dict.update(metrics)
        self.cpu_metric_input.append((self.metrics, yr_hat[0].detach().cpu(), yr[0].detach().cpu(), x_ref.expand_as(yr[0]).detach().cpu(), sample_rate, 'cpu'))

        # write examples
        if self.write_examples < 0 or paras[0]['index'] < self.write_examples:
            GS.test_setp_write_example(
                self=self,
                xr=x[:, self.ref_channel],
                yr=yr,
                yr_hat=yr_hat,
                sample_rate=sample_rate,
                paras=paras,
                result_dict=result_dict,
                wavname=wavname,
                exp_save_path=self.exp_save_path,
            )

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            autocast.__exit__(None, None, None)

        # return metrics, which will be collected, saved in test_epoch_end
        if 'metrics' in paras[0]:
            del paras[0]['metrics']  # remove circular reference
        result_dict['paras'] = paras[0]
        self.results.append(result_dict)

        # return result_dict

    def predict_step(self, batch: Union[Tensor, Tuple[Tensor, Tensor, Dict]], batch_idx: Optional[int] = None, dataloader_idx: Optional[int] = None) -> Tensor:
        """predict step on self.device, could be called dirctly or by PytorchLightning automatically using predict dataset
        Args:
            batch: x or (x, ys, paras). shape of x [B, C, T]

        Returns:
            Tensor: ys_hat, shape [B, Spk, T]
        """
        if isinstance(batch, Tensor):
            x, ys = batch, None
            yr = None
        else:
            x, ys, paras = batch
            yr = ys[:, :, self.ref_channel, :] if ys[0] is not None else None

        # forward & loss
        yr_hat = self.forward(x)

        if self.loss.is_scale_invariant_loss:
            x_ref = x[:, self.ref_channel, :]
            yr_hat = recover_scale(preds=yr_hat, mixture=x_ref, scale_src_together=True if self.loss.loss_func == neg_sa_sdr else False, norm_if_exceed_1=False)

        if yr is not None:  # reorder yr_hat if given yr
            _, perms = pit(preds=yr_hat, target=yr, metric_func=si_sdr, eval_func='max')
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
            monitor='val/' + self.loss.name,
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