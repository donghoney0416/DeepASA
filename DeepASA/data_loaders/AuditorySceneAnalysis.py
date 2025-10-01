import os
import random
from os.path import *
from pathlib import Path
from typing import *
from typing import Callable, List, Optional, Tuple

import numpy as np
import soundfile as sf
# from scipy.io.wavfile import read
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Dataset
import csv

from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler
from data_loaders.utils.rand import randint
import librosa
from natsort import natsorted


class AuditorySceneAnalysisDataset(Dataset):
    """The Auditory Scene Analysis dataset"""
    def __init__(
        self, 
        asa_dir: str,
        dataset: str,
        sample_rate: int = 16000,
        num_speakers: int = 5,
        audio_time_len: Optional[float] = None,
        n_classes: int = 13
    ) -> None:
        """The Auditory Scene Analysis dataset

        Args:
            asa_dir: directory of the dataset
            sample_rate: sample rate of the audio
            num_speakers: number of speakers
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
            n_classes: number of classes
        """
        super().__init__()

        self.asa_dir = str(Path(asa_dir).expanduser())
        self.wav_dir = Path(self.asa_dir) / dataset
        self.files = os.listdir(Path(self.wav_dir) / 'mixed')
        self.wav_files = [file for file in self.files if file.endswith(".wav")]
        assert len(self.wav_files) > 0, f"dir is empty or not exists: {self.asa_dir}"

        self.dataset = dataset
        self.sr = sample_rate
        self.audio_time_len = audio_time_len
        self.n_classes = n_classes

    def __getitem__(self, index_seed: Union[int, Tuple[int, int]]):
        if type(index_seed) == int:
            index = index_seed
            if self.dataset == 'tr':
                seed = random.randint(a=0, b=99999999)
            else:
                seed = index
        else:
            index, seed = index_seed
        g = torch.Generator()
        g.manual_seed(seed)
        front = self.wav_files[index].split('.')[0]
        # load audio signals
        mix, sr = sf.read(self.wav_dir / f'mixed/{front}.wav')
        reverb_dir = os.listdir(self.wav_dir / f'reverb/{front}')
        wav_files = [file for file in reverb_dir if file.endswith(".wav")]
        wav_files = natsorted(wav_files)
        # load files
        reverb_wavs = []
        anechoic_wavs = []
        for i in range(min(len(wav_files), 5)):  # load maximum 5 files
            reverb_wavs.append(sf.read(self.wav_dir / f'reverb/{front}' / wav_files[i])[0].T)
            anechoic_wavs.append(sf.read(self.wav_dir / f'anechoic/{front}' / wav_files[i])[0].T)
        # zero padding
        zeros_wav = np.zeros_like(reverb_wavs[0]) if reverb_wavs else None
        while len(reverb_wavs) < 5:
            reverb_wavs.append(zeros_wav)
            anechoic_wavs.append(zeros_wav)
        reverb = np.stack(reverb_wavs, axis=0)
        anechoic = np.stack(anechoic_wavs, axis=0)
        mix = mix.T

        # info
        info_file = f'info/{front}.csv'
        _output_dict = {}
        _fid = open(os.path.join(self.wav_dir, info_file), 'r')
        with open(os.path.join(self.wav_dir, info_file), 'r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            for row in csv_reader:
                _frame_ind = int(float(row[0]))
                if _frame_ind not in _output_dict: _output_dict[_frame_ind] = []
                if len(row) == 5 or len(row) == 6:
                    _output_dict[_frame_ind].append([int(float(row[1])), int(float(row[2])), float(row[3]), float(row[4])]) # src_idx, class, azi, ele
        _desc_file = self.convert_output_format_polar_to_cartesian(_output_dict)

        # Audio tagging
        self.label_len = int(self.audio_time_len * 10)
        cls_label = np.zeros(len(wav_files))  # (S, C)
        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < self.label_len:
                for active_event in active_event_list:
                    cls_label[active_event[0]] = active_event[1]
        # Zero-padding for cls_label (Ensure length is 5)
        zero_cls = -1 * np.ones(1)
        cls_label = np.concatenate([cls_label] + [zero_cls] * (5 - len(wav_files)), axis=0)
        # Sound event detection
        sed_label = np.zeros((len(wav_files), self.label_len, self.n_classes))  # (S, T, C)
        x_label = np.zeros((len(wav_files), self.label_len))  # (S, T)
        y_label = np.zeros((len(wav_files), self.label_len))  # (S, T)
        z_label = np.zeros((len(wav_files), self.label_len))  # (S, T)
        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < self.label_len:
                for active_event in active_event_list:
                    sed_label[active_event[0], frame_ind, active_event[1]] = 1
                    x_label[active_event[0], frame_ind] = active_event[2]
                    y_label[active_event[0], frame_ind] = active_event[3]
                    z_label[active_event[0], frame_ind] = active_event[4]
        # Zero-padding for sed_label (Ensure length is 5)
        zeros_sed = np.zeros((1, self.label_len, self.n_classes))
        sed_label = np.concatenate([sed_label] + [zeros_sed] * (5 - len(wav_files)), axis=0)
        # Sound event direction-of-arrival (DOA) labels
        doa_label = np.stack([x_label, y_label, z_label], axis=2)  # (S, T, XYZ)
        zeros_doa = np.zeros((1, self.label_len, 3))
        doa_label = np.concatenate([doa_label] + [zeros_doa] * (5 - len(wav_files)), axis=0)
        # Add silence class (-1 -> 14)
        # cls_label[cls_label == -1] = 13

        paras = {
            'index': index,
            'seed': seed,
            'wavname': self.files[index],
            'wavdir': str(self.wav_dir),
            'sample_rate': self.sr,
            'dataset': self.dataset,
            'audio_time_len': self.audio_time_len,
        }
        return torch.as_tensor(mix, dtype=torch.float32), torch.as_tensor(anechoic, dtype=torch.float32), torch.as_tensor(reverb, dtype=torch.float32), torch.as_tensor(len(wav_files), dtype=torch.int64), torch.as_tensor(cls_label, dtype=torch.long), torch.as_tensor(sed_label, dtype=torch.float32), torch.as_tensor(doa_label, dtype=torch.float32), paras

    def __len__(self):
        return len(self.files)

    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    ele_rad = (tmp_val[3]-90)*np.pi/180.
                    azi_rad = tmp_val[2]*np.pi/180
                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
        return out_dict


class ASADataModule(LightningDataModule):
    def __init__(
        self,
        asa_dir: str,
        sample_rate: int = 16000,
        num_speakers: int = 5,
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float]] = [4.0, 4.0, None],  # audio_time_len (seconds) for training, val, test.
        batch_size: List[int] = [1, 1],
        n_classes: int = 13,
        num_workers: int = 5,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int, int] = [None, 2, 3],  # random seeds for train, val and test sets
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = True,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.asa_dir = asa_dir
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers
        self.audio_time_len = audio_time_len
        self.n_classes = n_classes
        self.persistent_workers = persistent_workers

        rank_zero_info("dataset: AuditorySceneAnalysis")
        rank_zero_info(f'train/valid/test set: {sample_rate}, time length={audio_time_len}, {num_speakers}spk')

        self.batch_size = batch_size[0]
        self.batch_size_val = batch_size[1]
        self.batch_size_test = 1
        if len(batch_size) > 2:
            self.batch_size_test = batch_size[2]
        rank_zero_info(f'batch size: train={self.batch_size}; val={self.batch_size_val}; test={self.batch_size_test}')
        # assert self.batch_size_test == 1, "batch size for test should be 1 as the audios have different length"

        self.num_workers = num_workers

        self.collate_func_train = collate_func_train
        self.collate_func_val = collate_func_val
        self.collate_func_test = collate_func_test

        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        # if stage is not None and stage == 'test':
        #     audio_time_len = None
        # else:
        #     audio_time_len = self.audio_time_len[0]

        self.train = AuditorySceneAnalysisDataset(
            asa_dir=self.asa_dir,
            dataset='dev',
            audio_time_len=self.audio_time_len[0],
            sample_rate=self.sample_rate,
            n_classes=self.n_classes
        )
        self.val = AuditorySceneAnalysisDataset(
            asa_dir=self.asa_dir,
            dataset='test',
            audio_time_len=self.audio_time_len[1],
            sample_rate=self.sample_rate,
            n_classes=self.n_classes
        )
        self.test = AuditorySceneAnalysisDataset(
            asa_dir=self.asa_dir,
            dataset='test',
            audio_time_len=self.audio_time_len[2],
            sample_rate=self.sample_rate,
            n_classes=self.n_classes
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            sampler=MyDistributedSampler(self.train, seed=self.seeds[0], shuffle=True),
            batch_size=self.batch_size,
            collate_fn=self.collate_func_train,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            sampler=MyDistributedSampler(self.val, seed=self.seeds[1], shuffle=False),
            batch_size=self.batch_size_val,
            collate_fn=self.collate_func_val,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        # if self.test_set == 'test':
        #     dataset = self.test
        # elif self.test_set == 'val':
        #     dataset = self.val
        # else:  # train
        #     dataset = self.train

        return DataLoader(
            self.test,
            sampler=MyDistributedSampler(self.test, seed=self.seeds[2], shuffle=False),
            batch_size=self.batch_size_test,
            collate_fn=self.collate_func_test,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


if __name__ == '__main__':
    """python -m data_loaders.spatialized_wsj0_mix"""
    from argparse import ArgumentParser
    parser = ArgumentParser("")
    parser.add_argument('--sp_wsj0_dir', type=str, default='./datasets/spatialized-wsj0-mix')
    parser.add_argument('--version', type=str, default='min')
    parser.add_argument('--target', type=str, default='reverb')
    parser.add_argument('--sample_rate', type=int, default=8000)
    parser.add_argument('--gen_unprocessed', type=bool, default=True)
    parser.add_argument('--gen_target', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='dataset/sp_wsj')
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'val', 'test'])

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
