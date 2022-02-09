import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(samples, batch_first=True):
    audio, labels = zip(*samples)
    audio_pad = pad_sequence(audio, batch_first=batch_first)
    labels_pad = pad_sequence(labels, batch_first=batch_first)
    mask = [[int(j < labels[i].shape[-1]) for j in range(len(labels_pad[i]))] for i in range(len(samples))]

    return audio_pad, labels_pad, torch.tensor(mask).int()