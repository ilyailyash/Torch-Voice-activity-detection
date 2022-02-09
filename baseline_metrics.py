import os
from pathlib import Path

import torch
import webrtcvad
import numpy as np
from tqdm import tqdm
from glob import glob

from src.util.acoustic_utils import read_wave, Frame, frame_generator


def baseline_metrics(dataset_dir, vad_mode=1):

    noisy_files_list = glob(f'{dataset_dir}/*.wav', recursive=True)

    vad = webrtcvad.Vad(vad_mode)

    total_f1 = 0
    total_fpr = 0
    total_fnr = 0

    for i, (file_path) in tqdm(enumerate(noisy_files_list), total=len(noisy_files_list)):

        audio_b, sr, n_frames = read_wave(file_path)
        pred_labels = np.array([vad.is_speech(frame.bytes, sr) for frame in frame_generator(10, audio_b, sr)])

        file_path = Path(file_path)
        file_id = file_path.stem
        labels_dir = str(file_path.parent)+'-labels'

        labels_file_name = os.path.join(file_path.parents[1], labels_dir, f'{file_id}.pt')

        labels = torch.load(labels_file_name).numpy()

        TP = (pred_labels[labels == 1] == 1).sum()
        FP = (pred_labels[labels == 0] == 1).sum()
        TN = (pred_labels[labels == 0] == 0).sum()
        FN = (pred_labels[labels == 1] == 0).sum()

        if FP == TN == 0:
            total_fpr += 1
        else:
            total_fpr += FP/(FP+TN)

        total_fnr += FN/(FN+TP)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        total_f1 += 2*(precision*recall)/(precision+recall)

    total = len(noisy_files_list)
    return total_fpr/total,  total_fnr/total, total_f1/total


if __name__ == "__main__":
    dataset_dir = '/media/administrator/Data/train-clean-100/val/LibriSpeech/dev-noisy'
    fpr, fnr, f1 = baseline_metrics(dataset_dir)

    print(f'Baseline vad FPR = {fpr}, FNR = {fnr}, f1 = {f1}')

