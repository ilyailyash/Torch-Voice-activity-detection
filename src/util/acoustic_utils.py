import os
import wave
import contextlib
import random

import numpy as np
import torch
import torchaudio

torchaudio.set_audio_backend("sox_io")


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate, wf.getnframes()


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def compression_using_hyperbolic_tangent(mask, K=10, C=0.1):
    """
        (-inf, +inf) => [-K ~ K]
    """
    if torch.is_tensor(mask):
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - torch.exp(-C * mask)) / (1 + torch.exp(-C * mask))
    else:
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask



def complex_mul(noisy_r, noisy_i, mask_r, mask_i):
    r = noisy_r * mask_r - noisy_i * mask_i
    i = noisy_r * mask_i + noisy_i * mask_r
    return r, i


def stft(y, n_fft, hop_length, win_length, device='cpu', center=True, power=None):

    window = torch.hamming_window(n_fft).to(device)
    return torch.stft(y, n_fft, hop_length, win_length, window=window, center=center, onesided=False)


def istft(complex_tensor, n_fft, hop_length, win_length, device, length=None,
          use_mag_phase=False, center=True, power=None):
    if win_length is not None:
        window = torch.hamming_window(n_fft).to(device)
    else:
        win_length = None
        window = None

    if use_mag_phase:
        assert isinstance(complex_tensor, tuple) or isinstance(complex_tensor, list)
        mag, phase = complex_tensor
        complex_tensor = torch.stack([(mag * torch.cos(phase)), (mag * torch.sin(phase))], dim=-1)
    y = torch.istft(complex_tensor, n_fft, hop_length, win_length, window, length=length, center=center)

    return y


def mag_phase(complex_tensor):
    mag = (complex_tensor.pow(2.).sum(-1) + 1e-8).pow(0.5 * 1.0)
    phase = torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])
    return mag, phase


def norm_amplitude(y, scalar=None, eps=1e-6):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = np.sqrt(np.mean(y ** 2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scalar
    return y, rms, scalar


def is_clipped(y, clipping_threshold=0.999):
    return any(np.abs(y) > clipping_threshold)


def load_wav(file, sr=16000):
    if len(file) == 2:
        return file[-1]
    else:
        try:
            return torchaudio.load(os.path.abspath(os.path.expanduser(file)))[0][0].numpy()
        except Exception as e:
            path = os.path.abspath(os.path.expanduser(file))
            #if os.path.exists(path):
            #    os.remove(path)
            print('cant load file ' + path)
            return torchaudio.load('/media/administrator/Data/DNS-Challenge'
                                   '/datasets/clean/read_speech/book_00000_chp_0009_reader_06709_0.wav')[0][0].numpy()
        # return librosa.load(os.path.abspath(os.path.expanduser(file)), mono=False, sr=sr)[0]



def aligned_subsample(data_a, data_b, sub_sample_length):
    """
    Start from a random position and take a fixed-length segment from two speech samples

    Notes
        Only support one-dimensional speech signal (T,) and two-dimensional spectrogram signal (F, T)
    """
    assert data_a.shape == data_b.shape, "Inconsistent dataset size."

    dim = np.ndim(data_a)
    assert dim == 1 or dim == 2, "Only support 1D or 2D."

    if data_a.shape[-1] > sub_sample_length:
        length = data_a.shape[-1]
        start = np.random.randint(length - sub_sample_length + 1)
        end = start + sub_sample_length
        if dim == 1:
            return data_a[start:end], data_b[start:end]
        else:
            return data_a[:, start:end], data_b[:, start:end]
    elif data_a.shape[-1] == sub_sample_length:
        return data_a, data_b
    else:
        length = data_a.shape[-1]
        if dim == 1:
            return (
                np.append(data_a, np.zeros(sub_sample_length - length, dtype=np.float32)),
                np.append(data_b, np.zeros(sub_sample_length - length, dtype=np.float32))
            )
        else:
            return (
                np.append(data_a, np.zeros(shape=(data_a.shape[0], sub_sample_length - length), dtype=np.float32), axis=-1),
                np.append(data_b, np.zeros(shape=(data_a.shape[0], sub_sample_length - length), dtype=np.float32), axis=-1)
            )


def subsample(data, sub_sample_length):
    assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
    length = len(data)

    if length > sub_sample_length:
        start = np.random.randint(length - sub_sample_length)
        end = start + sub_sample_length
        data = data[start:end]
        assert len(data) == sub_sample_length
        return data
    elif length < sub_sample_length:
        data = np.append(data, np.zeros(sub_sample_length - length, dtype=np.float32))
        return data
    else:
        return data


def clean_noisy_subsample(noisy, clean, sub_sample_length):
    assert clean.dim() == 1, f"Only support 1D data. The dim is {clean.dim()}"
    assert noisy.dim() == 1, f"Only support 1D data. The dim is {noisy.dim()}"
    length = clean.shape[0]

    if length > sub_sample_length:
        start = random.randint(0, length - sub_sample_length)
        end = start + sub_sample_length
        clean = clean[start:end]
        noisy = noisy[start:end]
        assert clean.shape[0] == sub_sample_length
        return noisy, clean
    elif length < sub_sample_length:
        clean = torch.stack((clean, torch.zeros(sub_sample_length - length)))
        noisy = torch.stack((clean, torch.zeros(sub_sample_length - length)))
        return noisy, clean
    else:
        return noisy, clean


def noisy_clean_noise_subsample(noisy, clean, noise, sub_sample_length):
    assert clean.dim() == 1, f"Only support 1D data. The dim is {clean.dim()}"
    length = clean.shape[0]

    if length > sub_sample_length:
        start = random.randint(0, length - sub_sample_length)
        end = start + sub_sample_length
        clean = clean[start:end]
        noisy = noisy[start:end]
        noise = noise[start:end]
        assert clean.shape[0] == sub_sample_length
        return noisy, clean, noise
    elif length < sub_sample_length:
        clean = torch.stack((clean, torch.zeros(sub_sample_length - length)))
        noise = torch.stack((noise, torch.zeros(sub_sample_length - length)))
        noisy = torch.stack((clean, torch.zeros(sub_sample_length - length)))
        return noisy, clean, noise
    else:
        return noisy, clean, noise


def overlap_cat(chunk_list, dim=-1):

    overlap_output = []
    for i, chunk in enumerate(chunk_list):
        first_half, last_half = torch.split(chunk, chunk.size(-1) // 2, dim=dim)
        if i == 0:
            overlap_output += [first_half, last_half]
        else:
            overlap_output[-1] = (overlap_output[-1] + first_half) / 2
            overlap_output.append(last_half)

    overlap_output = torch.cat(overlap_output, dim=dim)
    return overlap_output


def activity_detector(audio, fs=16000, activity_threshold=0.13, target_level=-25, eps=1e-6):
    """
    Return the percentage of the time the audio signal is above an energy threshold

    Args:
        audio:
        fs:
        activity_threshold:
        target_level:
        eps:

    Returns:

    """
    audio, _, _ = tailor_dB_FS(audio, target_level)
    window_size = 50  # ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win ** 2) + eps)
        frame_energy_prob = 1. / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (1 - alpha_att)
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (1 - alpha_rel)

        if smoothed_energy_prob > activity_threshold:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def drop_sub_band(input, num_sub_batches=3):
    """
    To reduce the computational complexity of the sub_band sub model in the FullSubNet model.

    Args:
        input: [B, C, F, T]
        num_sub_batches:

    Notes:
        'batch_size' of the input should be divisible by the value of 'num_sub_batch'.
        If not, the frequencies corresponding to the last sub batch will not be well-trained.

    Returns:
        [B, C, F // num_sub_batches, T]
    """
    if num_sub_batches < 2:
        return input

    batch_size, _, n_freqs, _ = input.shape
    sub_batch_size = batch_size // num_sub_batches
    reminder = n_freqs % num_sub_batches

    output = []
    for idx in range(num_sub_batches):
        batch_indices = torch.arange(idx * sub_batch_size, (idx + 1) * sub_batch_size, device=input.device)
        freq_indices = torch.arange(
            idx + (reminder // 2),
            n_freqs - (reminder - reminder // 2),
            step=num_sub_batches,
            device=input.device
        )

        selected_sub_batch = torch.index_select(input, dim=0, index=batch_indices)
        selected_freqs = torch.index_select(selected_sub_batch, dim=2, index=freq_indices)
        output.append(selected_freqs)

    return torch.cat(output, dim=0)


if __name__ == '__main__':
    ipt = torch.rand(70, 1, 257, 200)
    print(drop_sub_band(ipt, 1).shape)
