import os

import numpy as np
from glob import glob
import struct

import webrtcvad
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

from src.util.acoustic_utils import norm_amplitude, read_wave, frame_generator


class VoiceBankDemandDataset(Dataset):
    def __init__(self, data_dir, sr=16000, sub_sample_length=None, hop_length=255, vad_mode=3, noise_proportion=0.9):
        self.clean_path = self.get_clean(data_dir)
        self.noisy_path = self.get_noisy(data_dir)
        self.sr = sr
        self.sub_sample_length = sub_sample_length
        self.hop_length = hop_length
        self.noise_proportion = noise_proportion
        self.vad = webrtcvad.Vad(vad_mode)

    def get_vad_lables(self, clean_path):
        audio_b, sr, n_frames = read_wave(clean_path)
        lables = np.array([self.vad.is_speech(frame.bytes, sr) for frame in frame_generator(10, audio_b, sr)])
        audio = np.array(struct.unpack('{n}h'.format(n=n_frames), audio_b))
        audio, _ = norm_amplitude(audio, self.scale)
        return audio, lables

    def get_clean(self, root):
        clean_dir = os.path.join(root, 'clean_testset_wav')
        filenames = glob(f'{clean_dir}/*.wav', recursive=True)
        return filenames

    def get_noisy(self, root):
        noisy_dir = os.path.join(root, 'noisy_testset_wav')
        filenames = glob(f'{noisy_dir}/*.wav', recursive=True)
        return filenames

    def padding(self, x):
        len_x = x.size(-1)
        pad_len = self.hop_length - len_x % self.hop_length
        x = F.pad(x, (0, pad_len))
        return x

    def normalize(self, x):
        return 2 * (x - x.min()) / (x.max() - x.min()) - 1

    def __len__(self):
        return len(self.noisy_path)

    def __getitem__(self, idx):

        audio, labels = self.get_vad_lables(self.clean_path[idx])

        if bool(np.random.random(1) < self.noise_proportion):
            audio, _ = torchaudio.load(self.noisy_path[idx])
        else:
            audio = torch.tensor(audio)

        labels = torch.tensor(labels)

        return audio.float(), labels.float()