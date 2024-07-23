import torch
import soundfile as sf
from vocos.pretrained import Vocos
from vocos.feature_extractors import EncodecFeatures, MelSpectrogramFeatures
import torchaudio
import numpy as np
import os
import glob

def from_model(cls, model_path: str, config_path: str) -> Vocos:
    """
    Class method to create a new Vocos saved model
    """
    model = cls.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(model.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in model.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model



model_dir = 'logs/lightning_logs/version_0/'

if not os.path.exists(model_dir + 'pytorch.bin'):
    checkpoint = model_dir + "checkpoints/last.ckpt"
    ckpt = torch.load(checkpoint)
    torch.save(ckpt['state_dict'], model_dir + 'pytorch.bin')


# vocos = Vocos.from_pretrained(model_dir)
vocos = from_model(
    cls=Vocos,
    model_path= model_dir + 'pytorch.bin',
    config_path= model_dir + 'model_config.yaml',
    )

# analysis by synthesis
audio, sr = torchaudio.load('data/wavs/LJ011-0256.wav')
mel_t = MelSpectrogramFeatures(sample_rate=22050, n_mels=80)
mel = mel_t(audio)
x = mel
audio = vocos.decode(x)
torchaudio.save('out.wav', audio, 22050, encoding='PCM_S')


print('done')
