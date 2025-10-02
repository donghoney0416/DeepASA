import json
import os
from typing import *
import wandb

import pytorch_lightning as pl
# import soundfile as sf
from scipy.io.wavfile import write
import torch
import torch.distributed as dist
from numpy import ndarray
from pandas import DataFrame
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch import Tensor

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from models.utils import MyJsonEncoder, tag_and_log_git_status
from models.utils.ensemble import ensemble
from models.utils.flops import write_FLOPs
from models.utils.metrics import (cal_metrics_functional, cal_pesq, recover_scale)
from models.utils.SELD_evaluation_metrics import *


def gather_tensor_ddp(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size() > 1:
        # 먼저 모양을 맞춰야 하므로 gather할 텐서의 크기 정보를 모두 모음
        local_shape = torch.tensor(tensor.shape, device=tensor.device)
        shape_list = [torch.zeros_like(local_shape) for _ in range(dist.get_world_size())]
        dist.all_gather(shape_list, local_shape)
        # 최대 크기로 패딩
        max_shape = torch.stack(shape_list).max(dim=0).values.tolist()
        padded = torch.zeros(*max_shape, dtype=tensor.dtype, device=tensor.device)
        slices = tuple(slice(0, s) for s in tensor.shape)
        padded[slices] = tensor
        # gather
        gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, padded)
        # 잘라서 원래 크기만큼만 concat
        tensors = [g[tuple(slice(0, s.item()) for s in shape)] for g, shape in zip(gathered, shape_list)]
        return torch.cat(tensors, dim=0)
    else:
        return tensor


def on_validation_epoch_end(self: pl.LightningModule, cpu_metric_input: List[Tuple[ndarray, ndarray, int]], N: int = 5) -> None:
    """calculate heavy metrics for every N epochs

    Args:
        self: LightningModule
        cpu_metric_input: the input list for cal_metrics_functional
        N: the number of epochs. Defaults to 5.
    """
    if self.current_epoch != 0 and self.current_epoch % N != (N - 1):
        cpu_metric_input.clear()
        return

    if len(cpu_metric_input) != 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
        num_thread = torch.multiprocessing.cpu_count() // (self.trainer.world_size * 2)
        p = torch.multiprocessing.Pool(min(num_thread, len(cpu_metric_input)))
        cpu_metrics = list(p.starmap(cal_metrics_functional, cpu_metric_input))
        p.close()
        p.join()

        for k in cpu_metric_input[0][0]:
            ms = list(filter(None, [m[0][k.lower()] for m in cpu_metrics]))
            if len(ms) > 0:
                self.log(f'val/{k}', sum(ms) / len(ms), sync_dist=True, batch_size=len(ms))
                self.val_wandb_log[f'val/{k}'] = ms

        cpu_metric_input.clear()

    if self.trainer.world_size > 1:
        dist.barrier()
        results_list = [None for _ in range(self.trainer.world_size)]
        dist.all_gather_object(results_list, self.val_wandb_log)
        # Dongheon Lee addition
        # 원래 GPU마다 local에 있던 값들
        features   = gather_tensor_ddp(torch.cat(self.features, dim=0))       # (B, T, D)
        sed_label  = gather_tensor_ddp(torch.cat(self.sed_label, dim=0))      # (B, T, C)
        doa_out    = gather_tensor_ddp(torch.cat(self.doa_out, dim=0))        # (B, T, 3)
        doa_label  = gather_tensor_ddp(torch.cat(self.doa_label, dim=0))      # (B, T, 3)
        active_mask = gather_tensor_ddp(torch.cat(self.onoff_out, dim=0))     # (B, T)
        # Calculate metrics
        cls_binary = torch.zeros_like(features).scatter_(-1, torch.argmax(features, dim=-1, keepdim=True), 1)
        sed_out = active_mask.unsqueeze(-1) * cls_binary.unsqueeze(2)
        er, f1, le, lr, sc, f1_c, le_c, lr_c = SELD_metrics_c(sed_out, sed_label, doa_out, doa_label, per_class=True)   
        if self.trainer.is_global_zero: 
            class_names = ["Female speech", "Male speech", "Clapping", "Telephone", "Laughter", "Domestic sound", "Walk and footsteps", "Door", "Music", "Musical instrument", "Water tap", "Bell", "Knock"]
            for i, (f1_, le_, lr_) in enumerate(zip(f1_c, le_c, lr_c)):
                class_name = class_names[i] if i < len(class_names) else f"Class {i}"
                print(f"{class_name:<24} | F1: {f1_:.2f} | LE: {le_:.2f} | LR: {lr_:.2f}")
        # Wandb logging
        if self.trainer.sanity_checking:
            self.val_wandb_log.update({'ER': [], 'F1': [], 'LE': [], 'LR': [], 'SELD': []})
        # else:
            # self.val_wandb_log['ER'].append(er)
            # self.val_wandb_log['F1'].append(f1)
            # self.val_wandb_log['LE'].append(le)
            # self.val_wandb_log['LR'].append(lr)
            # self.val_wandb_log['SELD'].append(sc)
        # classification evaluation
        self.log('ER', er, sync_dist=True, batch_size=2)
        self.log('F1', f1, sync_dist=True, batch_size=2)
        # DoA estimation evaluation
        self.log('LE', le, sync_dist=True, batch_size=2)
        self.log('LR', lr, sync_dist=True, batch_size=2)
        # SELD score
        self.log('SELD', sc, sync_dist=True, batch_size=2)
        # logging
        val_metric = {'ER': er, 'F1': f1, 'LE': le, 'LR': lr, 'SC': sc}[self.val_metric]
        self.log('val/metric', val_metric, sync_dist=True, batch_size=2)  # log val/metric for checkpoint picking

        merged_results = {}

        for results in results_list:
            for k, v in results.items():
                if isinstance(v, list):
                    if k in merged_results:
                        merged_results[k].extend(v)
                    elif len(v) > 0:
                        merged_results[k] = v
                elif k == 'epoch' or k == 'learning_rate':
                    merged_results[k] = v
    
    # initialization
    self.features = []
    self.sed_label = []
    self.doa_out = []
    self.doa_label = []
    self.onoff_out = []

    # if self.trainer.is_global_zero and not self.trainer.sanity_checking:
    #     upload_wandb_log = {'val/epoch': self.current_epoch}
    #     for k, v in merged_results.items():
    #         if not isinstance(v, list):
    #             upload_wandb_log[f'val/{k}'] = v
    #         else:
    #             upload_wandb_log[f'val/{k}'] = sum(v) / len(v)

    #     wandb.log(upload_wandb_log)

    # for k, v in self.val_wandb_log.items():
    #     if isinstance(v, list):
    #         self.val_wandb_log[k] = []
    # return upload_wandb_log


def on_test_epoch_end(self: pl.LightningModule, results: List[Dict[str, Any]], cpu_metric_input: List, exp_save_path: str):
    """ calculate cpu metrics on CPU, collect results, save results to file

    Args:
        self: LightningModule
        results: the result list
        cpu_metric_input: the input list for cal_metrics_functional
        exp_save_path: the path to save result file
    """
    # calculate metrics, input_metrics, improve_metrics on CPU using multiprocessing to speed up
    torch.multiprocessing.set_sharing_strategy('file_system')
    num_thread = torch.multiprocessing.cpu_count() // (self.trainer.world_size * 2)
    p = torch.multiprocessing.Pool(min(num_thread, len(cpu_metric_input)))
    cpu_metrics = list(p.starmap(cal_metrics_functional, cpu_metric_input))
    p.close()
    p.join()

    for i, m in enumerate(cpu_metrics):
        metrics, input_metrics, imp_metrics = m
        results[i].update(input_metrics)
        results[i].update(imp_metrics)
        results[i].update(metrics)

        # # Extract class index (argmax)
        # cls_label_idx = metrics.get('cls_label')  # actual class index
        # cls_out_idx = metrics.get('cls_out')  # predicted class index (argmax)

        # # Extract direction of arrival (DOA) labels and outputs
        # doa_label_traj = metrics.get('doa_labe')  # shape (40, 3)
        # doa_out_traj = metrics.get('doa_out')  # shape (40, 3)
        # le = metrics.get('localization_error')

        # # Save additional metrics to the result
        # results[i].update({
        #     'cls_label_idx': cls_label_idx,  # Class label (ground truth)
        #     'cls_out_idx': cls_out_idx,  # Predicted class (argmax)
        #     'doa_label_traj': doa_label_traj,  # Actual DOA trajectory
        #     'doa_out_traj': doa_out_traj,  # Predicted DOA trajectory
        #     'localization_error': le  # Localization error (LE)
        # })

    # gather results from all GPUs
    import torch.distributed as dist

    if self.trainer.world_size > 1:
        dist.barrier()
        results_list = [None for _ in range(self.trainer.world_size)]
        dist.all_gather_object(results_list, results)  # gather results from all GPUs
    else:
        results_list  = [results]

    RESULT_SAMPLE_KEY = ['wavname', 'cls_out', 'cls_label', 'sed_out', 'sed_label', 'doa_out', 'onoff_out', 'act_out', 'doa_label', 'onoff_label', 'features']

    if self.trainer.is_global_zero:
        device = f'cuda:{dist.get_rank()}'
        results_sample_, results_segment_ = [], []
        for rr in results_list:
            for rs in rr:
                rst_spl, rst_seg = {}, {}
                for k, v in rs.items():
                    if k in RESULT_SAMPLE_KEY:
                        rst_spl[k] = v if not isinstance(v, Tensor) else v.to(device)
                    else:
                        rst_seg[k] = v if not isinstance(v, Tensor) else v.to(device)
                results_sample_.append(rst_spl)
                results_segment_.append(rst_seg)

        results_sample = {}
        for rst_spl in results_sample_:
            added_waveform = rst_spl['wavname'] in results_sample.keys()
            if not added_waveform:
                results_sample[rst_spl['wavname']] = {}
                for k, v in rst_spl.items():
                    if k == 'wavname':
                        continue
                    results_sample[rst_spl['wavname']][k] = [v]
            else:
                for k, v in rst_spl.items():
                    if k == 'wavname':
                        continue
                    results_sample[rst_spl['wavname']][k].append(v)

        # results_segment = {}
        # for rst_seg in results_segment_:
        #     for k, v in rst_seg.items():
        #         if k in results_segment:
        #             results_segment[k].extend(v)
        #         elif len(v) > 0:
        #             results_segment[k] = v
        results = results_segment_

    # save collected data on 0-th gpu
    if self.trainer.is_global_zero:
        # Save individual results
        import datetime
        x = datetime.datetime.now()
        dtstr = x.strftime('%Y%m%d_%H%M%S.%f')
        path = os.path.join(exp_save_path, 'results_{}.json'.format(dtstr))
        # write results to json
        f = open(path, 'w', encoding='utf-8')
        json.dump(results, f, indent=4, cls=MyJsonEncoder)
        f.close()

        # write mean to json
        df = DataFrame(results)
        df.mean(numeric_only=True).to_json(os.path.join(exp_save_path, 'results_mean.json'), indent=4)
        self.print('results: ', os.path.join(exp_save_path, 'results_mean.json'), ' ', path)

    if self.trainer.is_global_zero:
        return results_sample


def on_predict_batch_end(
    self: pl.LightningModule,
    outputs: Optional[Any],
    batch: Any,
) -> None:
    """save predicted results to `log_dir/examples`

    Args:
        self: LightningModule
        outputs: _description_
        batch: _description_
    """
    save_dir = self.trainer.logger.log_dir + '/' + 'examples'
    os.makedirs(save_dir, exist_ok=True)

    if not isinstance(batch, Tensor):
        _, _, paras = batch
        if 'saveto' in paras[0]:
            for b in range(len(paras)):
                saveto = paras[b]['saveto']
                if isinstance(saveto, str):
                    saveto = [saveto]
                assert isinstance(saveto, list), type(saveto)
                for spk, spk_saveto in enumerate(saveto):
                    y = outputs[b][spk]
                    assert len(y.shape) == 1, y.shape
                    save_path = save_dir + '/' + spk_saveto
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    write(save_path, samplerate=paras[b]['sample_rate'], data=y.detach().cpu().numpy())


def on_load_checkpoint(
    self: pl.LightningModule,
    checkpoint: Dict[str, Any],
    ensemble_opts: Union[int, str, List[str], Literal[None]] = None,
    compile: bool = True,
) -> None:
    """load checkpoint

    Args:
        self: LightningModule
        checkpoint: the loaded weights
        ensemble_opts: opts for ensemble. Defaults to None.
        compile: whether the checkpoint is a compiled one. Defaults to True.
    """
    if ensemble_opts:
        ckpt = self.trainer.ckpt_path
        ckpts, state_dict = ensemble(opts=ensemble_opts, ckpt=ckpt)
        self.print(f'rank {self.trainer.local_rank}/{self.trainer.world_size}, ensemble {ensemble_opts}: {ckpts}')
        checkpoint['state_dict'] = state_dict

    # rename weights for removing _orig_mod in name
    if compile == False:
        state_dict = checkpoint['state_dict']
        state_dict_new = dict()
        for k, v, in state_dict.items():
            state_dict_new[k.replace('_orig_mod.', '')] = v
        checkpoint['state_dict'] = state_dict_new
    return super(pl.LightningModule, self).on_load_checkpoint(checkpoint)


def on_train_start(self: pl.LightningModule, exp_name: str, model_name: str, num_chns: int, nfft: int, model_class_path: str = None):
    """ 1) add git tags/write requirements for better change tracking; 2) write model architecture to file; 3) measure the model FLOPs

    Args:
        self: LightningModule
        exp_name: `notag` or exp_name, add git tag e.g. 'model_name_v10' if exp_name!='notag'
        model_name: the model name
        num_chns: the number of channels for FLOPs test
        nfft: the number of fft points
        model_class_path: the path to import the self
    """
    if self.current_epoch == 0:
        if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir') and 'notag' not in exp_name:
            # add git tags for better change tracking
            # note: if change self.logger.log_dir to self.trainer.log_dir, the training will stuck on multi-gpu training
            tag_and_log_git_status(self.logger.log_dir + '/git.out', self.logger.version, exp_name, model_name=model_name)

        # if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir'):
        #     # write model architecture to file
        #     with open(self.logger.log_dir + '/model.txt', 'a') as f:
        #         f.write(str(self))
        #         f.write('\n\n\n')
        #     # measure the model FLOPs, the num_chns here only means the original channels
        #     write_FLOPs(model=self, save_dir=self.logger.log_dir, num_chns=num_chns, nfft=nfft, model_class_path=model_class_path)


def configure_optimizers(
    self: pl.LightningModule,
    optimizer: str,
    optimizer_kwargs: Dict[str, Any],
    monitor: str = 'val/loss',
    lr_scheduler: str = None,
    lr_scheduler_kwargs: Dict[str, Any] = None,
):
    """configure optimizer and lr_scheduler"""
    if optimizer == 'Adam' and self.trainer.precision == '16-mixed':
        if 'eps' not in optimizer_kwargs:
            optimizer_kwargs['eps'] = 1e-4  # according to https://discuss.pytorch.org/t/adam-half-precision-nans/1765
            rank_zero_info('setting the eps of Adam to 1e-4 for FP16 mixed precision training')
        else:
            allowed_minimum = torch.finfo(torch.float16).eps
            assert optimizer_kwargs['eps'] >= allowed_minimum, f"You should specify an eps greater than the allowed minimum of the FP16 precision: {optimizer_kwargs['eps']} {allowed_minimum}"

    # optimizer = getattr(torch.optim, optimizer)(self.parameters(), **optimizer_kwargs)
    optimizer = getattr(torch.optim, optimizer)(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_kwargs)

    if lr_scheduler is not None and len(lr_scheduler) > 0:
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler(optimizer, **lr_scheduler_kwargs),
                'monitor': monitor,
            }
        }
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1, (1/2) ** epoch))
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(8, 2 ** epoch))
        # return {
        #         'optimizer': optimizer,
        #         'lr_scheduler': {
        #             'scheduler': lr_scheduler,
        #             'monitor': monitor,
        #         }
        #     }
    else:
        return optimizer


def test_setp_write_example(self, xr: Tensor, yr: Tensor, yr_hat: Tensor, sample_rate: int, paras: Dict[str, Any], result_dict: Dict[str, Any], wavname: str, exp_save_path: str):
    """
    Args:
        xr: [B,T] - Mixed waveform
        yr: [B,Spk,T] - Ground truth for each speaker
        yr_hat: [B,Spk,T] - Predicted waveform for each speaker
        sample_rate: Sample rate of the audio
        paras: Hyperparameters or additional configuration
        result_dict: Results, metrics, etc.
        wavname: Filename for saving the example
        exp_save_path: Path to save the results
    """

    # write examples
    if yr is not None:
        abs_max = max(torch.max(torch.abs(xr[0, ...])), torch.max(torch.abs(yr[0, ...])))
    else:
        abs_max = torch.max(torch.abs(xr[0, ...]))

    def write_wav(wav_path: str, wav: torch.Tensor, norm_to: torch.Tensor = None):
        # make sure wav doesn't have illegal values (abs greater than 1)
        if norm_to:
            wav = wav / torch.max(torch.abs(wav)) * norm_to
        if abs_max > 1:
            wav /= abs_max
        abs_max_wav = torch.max(torch.abs(wav))
        if abs_max_wav > 1:
            import warnings
            warnings.warn(f"abs_max_wav > 1, {abs_max_wav}")
            wav /= abs_max_wav
        write(wav_path, sample_rate, wav.detach().cpu().numpy())

    pattern = '.'.join(wavname.split('.')[:-1]) + '{name}'  # remove .wav in wavname
    example_dir = os.path.join(exp_save_path, 'examples', str(paras[0]['index']))
    os.makedirs(example_dir, exist_ok=True)
    
    # save preds and targets for each speaker
    for i in range(yr_hat.shape[1]):
        # write ys (ground truth)
        if yr is not None:
            wav_path = os.path.join(example_dir, pattern.format(name=f"_spk{i+1}.wav"))
            write_wav(wav_path=wav_path, wav=yr[0, i])
        # write ys_hat (predicted)
        wav_path = os.path.join(example_dir, pattern.format(name=f"_spk{i+1}_p.wav"))
        write_wav(wav_path=wav_path, wav=yr_hat[0, i])  # , norm_to=ys[0, i].abs().max())

    # write mix
    wav_path = os.path.join(example_dir, pattern.format(name=f"_mix.wav"))
    write_wav(wav_path=wav_path, wav=xr[0, :])

    # Add cls_label, cls_out, doa_label, doa_out, and localization error to result_dict
    # Assuming that the relevant metrics are available in result_dict:
    cls_label_idx = result_dict.get('cls_label')
    cls_out_idx = result_dict.get('cls_out')
    doa_label_traj = result_dict.get('doa_label')
    doa_out_traj = result_dict.get('doa_out')
    le = result_dict.get('localization_error')

    # Add these values to result_dict
    result_dict.update({
        'cls_label': cls_label_idx[0],  # Actual class label index
        'cls_out': cls_out_idx[0],  # Predicted class label index
        'doa_label': doa_label_traj[0],  # Actual DOA trajectory
        'doa_out': doa_out_traj[0],  # Predicted DOA trajectory
        'localization_error': le  # Localization error (LE)
    })

    # Write paras & results to JSON file
    f = open(os.path.join(example_dir, pattern.format(name=f"_paras.json")), 'w', encoding='utf-8')
    paras[0]['metrics'] = result_dict  # Add metrics to paras
    json.dump(paras[0], f, indent=4, cls=MyJsonEncoder)
    f.close()