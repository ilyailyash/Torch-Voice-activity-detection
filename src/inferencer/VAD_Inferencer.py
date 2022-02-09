import os

import torch
import numpy as np

from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram
from src.common.inferencer import BaseInferencer


def cumulative_norm(input):
    eps = 1e-10
    device = input.device
    data_type = input.dtype
    n_dim = input.ndim

    assert n_dim in (3, 4)

    if n_dim == 3:
        n_channels = 1
        batch_size, n_freqs, n_frames = input.size()
    else:
        batch_size, n_channels, n_freqs, n_frames = input.size()
        input = input.reshape(batch_size * n_channels, n_freqs, n_frames)

    step_sum = torch.sum(input, dim=1)  # [B, T]
    step_pow_sum = torch.sum(torch.square(input), dim=1)

    cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
    cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

    entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
    entry_count = entry_count.reshape(1, n_frames)  # [1, T]
    entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

    cum_mean = cumulative_sum / entry_count  # B, T
    cum_var = (cumulative_pow_sum - 2 * cum_mean * cumulative_sum) / entry_count + cum_mean.pow(2)  # B, T
    cum_std = (cum_var + eps).sqrt()  # B, T

    cum_mean = cum_mean.reshape(batch_size * n_channels, 1, n_frames)
    cum_std = cum_std.reshape(batch_size * n_channels, 1, n_frames)

    x = (input - cum_mean) / cum_std

    if n_dim == 4:
        x = x.reshape(batch_size, n_channels, n_freqs, n_frames)

    return x


class Inferencer(BaseInferencer):
    def __init__(self, config, checkpoint_path, output_dir):
        super().__init__(config, checkpoint_path, output_dir)

        n_fft = config["acoustic"]["n_fft"]
        hop_length = config["acoustic"]["hop_length"]
        win_length = config["acoustic"]["win_length"]
        center = config["acoustic"]["center"]
        n_mel = config["acoustic"]["n_mel"]

        self.mel_spectrogram = MelSpectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center,
                                              n_mels=n_mel)
        self.mel_spectrogram.to(self.device)

    @torch.no_grad()
    def mel(self, noisy, inference_args):

        sr = inference_args['sr']
        out_len = int(noisy.shape[-1] // (sr/100))

        noisy_spec = self.mel_spectrogram(noisy)

        pred_scores = self.model(noisy_spec.unsqueeze(1))[0, :out_len]

        return pred_scores.cpu().numpy()

    @torch.no_grad()
    def __call__(self):
        inference_type = self.inference_config["type"]
        assert inference_type in dir(self), f"Not implemented Inferencer type: {inference_type}"

        inference_args = self.inference_config["args"]

        with open(os.path.join(self.scores_dir, 'eer.txt'), 'w') as f:
            for noisy, name in tqdm(self.dataloader, desc="Inference"):
                assert len(name) == 1, "The batch size of inference stage must 1."
                name = name[0]

                scores = getattr(self, inference_type)(noisy.to(self.device), inference_args)

                line = name + ', [' + ','.join([str(s) for s in scores]) + ']\n'
                f.write(line)

        with open(os.path.join(self.scores_dir, 'thresholds.txt'), 'w') as f:
            for key, val in self.thresholds.items():
                f.write(f'{key} = {val}\n')




if __name__ == '__main__':
    a = torch.rand(10, 2, 161, 200)
    print(cumulative_norm(a).shape)
