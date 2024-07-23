from vocos.feature_extractors import MelSpectrogramFeatures
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch
import glob
from tqdm import tqdm
import os

class CustomDataset(Dataset):
    def __init__(self, audio_paths):
        self.audio_paths = audio_paths
        self.mel_spec = MelSpectrogramFeatures(sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256, f_min=0.0, f_max=8000.0)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        audio, _ = torchaudio.load(audio_path)
        mel = self.mel_spec(audio)
        mel = torch.squeeze(mel)

        filename = audio_path.replace('/wavs/','/mels_vocos/').replace('.wav', '.pt')
        torch.save(mel, filename)

        return 'nana'

audio_dir = 'data/wavs'

os.makedirs(audio_dir+'/mels_vocos', exist_ok=True)

audio_paths = glob.glob(audio_dir + '/wavs/*.wav')
custom_dataset = CustomDataset(audio_paths)

data_loader = DataLoader(dataset=custom_dataset, batch_size=3, shuffle=False)
for batch in tqdm(data_loader):
    audio_dir

print('done')
