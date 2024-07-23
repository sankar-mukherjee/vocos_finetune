from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from vocos.modules import safe_log

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int
    teacher_forced_dir: str
    teacher_forced: bool = False


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        if cfg.teacher_forced:
            dataset = VocosDataset_teacherForced(cfg, train=train)
        else:
            dataset = VocosDataset(cfg, train=train)

        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        return y[0]


class VocosDataset_teacherForced(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train
        self.teacher_forced_dir = cfg.teacher_forced_dir

        self.hop_size = 256
        self.frames_per_seg = int(np.ceil(self.num_samples / self.hop_size)) + 1


    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        y, sr = torchaudio.load(audio_path)

        mel_path = self.teacher_forced_dir + '/' + audio_path.split('/')[-1].replace('.wav', '.pt')
        mel = torch.squeeze(torch.load(mel_path))


        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)

        # Fixed size
        if y.size(-1) < self.num_samples:
            y = F.pad(y, (0, self.num_samples - y.size(-1)))
        if mel.size(-1) < self.frames_per_seg:
            mel = F.pad(mel, (0, self.frames_per_seg - mel.size(-1)))

        if self.train:
            high = mel.size(-1) - self.frames_per_seg
            mel_start = 0 if high == 0 else np.random.randint(low=0, high=mel.size(-1) - self.frames_per_seg)
            
            mel = mel[:, mel_start:mel_start + self.frames_per_seg]
            a = mel_start * self.hop_size
            b = (mel_start + self.frames_per_seg-1) * self.hop_size
            y = y[:, a:b]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]
            mel = mel[:, : self.frames_per_seg]

        # checks
        if y.size(-1) < self.num_samples:
            y = F.pad(y, (0, self.num_samples - y.size(-1)))
        if mel.size(-1) < self.frames_per_seg:
            mel = F.pad(mel, (0, self.frames_per_seg - mel.size(-1)))

        assert y.size(-1) == self.num_samples, f'bad audio {y.shape, mel.shape, mel_path}'
        assert mel.size(-1) == self.frames_per_seg, f'bad mel {y.shape, mel.shape, mel_path}'

        if not y.size(-1) == self.num_samples:
            print(y.shape)

        if not mel.size(-1) == self.frames_per_seg:
            print(mel.shape)

        return y[0], mel
