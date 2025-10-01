import os
import random
from os.path import *
from pathlib import Path
from typing import *
from typing import Callable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Dataset
import csv

from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler
from data_loaders.utils.rand import randint
import librosa
import pandas as pd
from natsort import natsorted


class SpatializedWSJMixDataset(Dataset):
    """The Spatialized WSJ0-2/3Mix dataset"""
    def __init__(
        self,
        sp_wsj0_dir: str,
        dataset: str,
        version: str = 'min',
        target: str = 'reverb',
        audio_time_len: Optional[float] = 4,
        sample_rate: int = 16000,
        num_speakers: int = 5,
        label_len: int = 40,
        n_classes: int = 13
    ) -> None:
        """The Spatialized WSJ-2/3Mix dataset

        Args:
            sp_wsj0_dir: a dir contains [2speakers_reverb]
            dataset: tr, cv, tt
            target: anechoic or reverb
            version: min or max
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
        """
        super().__init__()
        assert sample_rate in [8000, 16000], sample_rate
        assert dataset in ['train', 'dev', 'test'], dataset
        assert num_speakers in [2, 3, 4, 5], num_speakers

        self.sp_wsj0_dir = str(Path(sp_wsj0_dir).expanduser())
        self.wav_dir = Path(self.sp_wsj0_dir) / dataset
        self.files = os.listdir(Path(self.wav_dir) / 'mixed')
        self.wav_files = [file for file in self.files if file.endswith(".wav")]
        # w/o augmentation
        # self.wav_files = [file for file in self.files if file.endswith(".wav") and (file.startswith("fold1") or file.startswith("fold2") or file.startswith("fold3") or file.startswith("fold4") or file.startswith("fold6"))]
    
        assert len(self.wav_files) > 0, f"dir is empty or not exists: {self.sp_wsj0_dir}"

        self.version = version
        self.dataset = dataset
        self.target = target
        self.audio_time_len = audio_time_len
        self.n_spk = num_speakers
        self.label_len = label_len
        self.n_classes = n_classes
        self.sr = sample_rate

    def __getitem__(self, index_seed: Union[int, Tuple[int, int]]):
        if type(index_seed) == int:
            index = index_seed % len(self.wav_files)
            if self.dataset == 'tr':
                seed = random.randint(a=0, b=99999999)
            else:
                seed = index
        else:
            index, seed = index_seed
            index = index % len(self.wav_files)
        g = torch.Generator()
        g.manual_seed(seed)
        front = self.wav_files[index].split('.')[0]
        # mixed
        mix, sr = sf.read(self.wav_dir / f'mixed/{front}.wav')
        mix = mix.T

        #  pad or cut signals
        T = mix.shape[-1]
        start = 0
        frames = int(sr * self.audio_time_len)
        if T < frames:
            mix = np.pad(mix, pad_width=((0, 0), (0, frames - T)), mode='constant', constant_values=0)
        elif T > frames:
            start = randint(g, low=0, high=T - frames)
            mix = mix[:, start:start + frames]
        
        info_file = f'info/{front}.csv'
        _output_dict = {}
        _fid = open(os.path.join(self.wav_dir, info_file), 'r')
        with open(os.path.join(self.wav_dir, info_file), 'r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            for row in csv_reader:
                _frame_ind = int(float(row[0]))-1
                if _frame_ind not in _output_dict: _output_dict[_frame_ind] = []
                if len(row) == 6:
                    _output_dict[_frame_ind].append([int(float(row[5]))-1, int(float(row[1])), float(row[3]), float(row[4])]) # src_idx, class, azi, ele
                elif len(row) == 5:
                    _output_dict[_frame_ind].append([int(float(row[1])), int(float(row[2])), float(row[3]), float(row[4])]) # src_idx, class, azi, ele
        _desc_file = self.convert_output_format_polar_to_cartesian(_output_dict)

        cls_label = -1 * np.ones(self.n_spk)        
        sed_label = np.zeros((self.label_len, self.n_spk, self.n_classes))      # (T, S, C)
        x_label = np.zeros((self.label_len, self.n_spk))       # (T, S)
        y_label = np.zeros((self.label_len, self.n_spk))       # (T, S)
        z_label = np.zeros((self.label_len, self.n_spk))       # (T, S)
        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < self.label_len:
                for active_event in active_event_list:
                    if active_event[0] < self.n_spk:
                        cls_label[active_event[0]] = active_event[1]
                        sed_label[frame_ind, active_event[0], active_event[1]] = 1
                        x_label[frame_ind, active_event[0]] = active_event[2]
                        y_label[frame_ind, active_event[0]] = active_event[3]
                        z_label[frame_ind, active_event[0]] = active_event[4]
        doa_label = np.stack([x_label, y_label, z_label], axis=2)   # (T, S, XYZ)
        n_spk = np.sum(cls_label != -1)
        # Add silence class (-1 -> 14)
        # cls_label[cls_label == -1] = 13

        paras = {
            'index': index,
            'seed': seed,
            'wavname': self.files[index],
            'wavdir': str(self.wav_dir),
            'sample_rate': self.sr,
            'dataset': self.dataset,
            'target': self.target,
            'version': self.version,
            'audio_time_len': self.audio_time_len,
            'start': start,
        }

        return torch.as_tensor(mix, dtype=torch.float32), \
            torch.as_tensor(cls_label, dtype=torch.long), \
            torch.as_tensor(sed_label, dtype=torch.float32).transpose(0, 1), \
            torch.as_tensor(doa_label, dtype=torch.float32).transpose(0, 1), \
            torch.as_tensor(n_spk, dtype=torch.int64), paras

    def __len__(self):
        return len(self.files)

    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    ele_rad = tmp_val[3]*np.pi/180
                    azi_rad = tmp_val[2]*np.pi/180
                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
        return out_dict


class DCASE2023DataModule(LightningDataModule):
    def __init__(
        self,
        sp_wsj0_dir: str,
        version: str = 'min',
        target: str = 'reverb',
        sample_rate: int = 16000,
        num_speakers: int = 4,
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float]] = [4.0, 4.0, 4.0],  # audio_time_len (seconds) for training, val, test.
        batch_size: List[int] = [1, 1],
        label_len: int = 40,
        n_classes: int = 13,
        test_set: str = 'test',  # the dataset to test: train, val, test
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
        self.sp_wsj0_dir = sp_wsj0_dir
        self.version = version
        self.target = target
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers
        self.audio_time_len = audio_time_len
        self.label_len = label_len
        self.n_classes = n_classes
        self.persistent_workers = persistent_workers
        self.test_set = test_set

        rank_zero_info("dataset: SpatializedWSJ0Mix")
        rank_zero_info(f'train/valid/test set: {version} {target} {sample_rate}, time length={audio_time_len}, {num_speakers}spk')
        # assert audio_time_len[2] == None, 'test audio time length should be None'

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
        if stage is not None and stage == 'test':
            audio_time_len = None
        else:
            audio_time_len = self.audio_time_len[0]

        self.train = SpatializedWSJMixDataset(
            sp_wsj0_dir=self.sp_wsj0_dir,
            dataset='test',
            version=self.version,
            target=self.target,
            audio_time_len=audio_time_len,
            sample_rate=self.sample_rate,
            label_len=self.label_len,
            n_classes=self.n_classes
        )
        self.val = SpatializedWSJMixDataset(
            sp_wsj0_dir=self.sp_wsj0_dir,
            dataset='train',
            version=self.version,
            target=self.target,
            audio_time_len=self.audio_time_len[1],
            sample_rate=self.sample_rate,
            label_len=self.label_len,
            n_classes=self.n_classes
        )
        self.test = SpatializedWSJMixDataset(
            sp_wsj0_dir=self.sp_wsj0_dir,
            dataset='test',
            version=self.version,
            target=self.target,
            audio_time_len=self.audio_time_len[2],
            sample_rate=self.sample_rate,
            label_len=self.label_len,
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
        if self.test_set == 'test':
            dataset = self.test
        elif self.test_set == 'val':
            dataset = self.val
        else:  # train
            dataset = self.train

        return DataLoader(
            dataset,
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

    if not args.gen_unprocessed and not args.gen_target:
        exit()

    datamodule = SpatializedWSJ0MixDataModule(args.sp_wsj0_dir, args.version, target=args.target, sample_rate=args.sample_rate, batch_size=[1, 1], num_workers=1)
    datamodule.setup()
    if args.dataset == 'train':
        dataloader = datamodule.train_dataloader()
    elif args.dataset == 'val':
        dataloader = datamodule.val_dataloader()
    else:
        assert args.dataset == 'test'
        dataloader = datamodule.test_dataloader()

    for idx, (mix, tar, paras) in enumerate(dataloader):
        # write target to dir
        print(mix.shape, tar.shape, paras)

        if idx > 10:
            continue

        if args.gen_target:
            tar_path = Path(f"{args.save_dir}/{args.target}/{args.dataset}").expanduser()
            tar_path.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                # assert np.max(np.abs(tar[0, i, 0, :].numpy())) <= 1
                sp = tar_path / (paras[0]['wavname'] + f'_spk{i}.wav')
                if not sp.exists():
                    sf.write(sp, tar[0, i, 0, :].numpy(), samplerate=paras[0]['sample_rate'])

        # write unprocessed's 0-th channel
        if args.gen_unprocessed:
            tar_path = Path(f"{args.save_dir}/unprocessed/{args.dataset}").expanduser()
            tar_path.mkdir(parents=True, exist_ok=True)
            # assert np.max(np.abs(mix[0, 0, :].numpy())) <= 1
            sp = tar_path / (paras[0]['wavname'])
            if not sp.exists():
                sf.write(sp, mix[0, 0, :].numpy(), samplerate=paras[0]['sample_rate'])
