import os
from pathlib import Path

import torch
import webrtcvad
import numpy as np
from tqdm import tqdm
from glob import glob

from src.util.acoustic_utils import read_wave, Frame, frame_generator
from src.util.metrics import get_f1


def baseline_metrics(dataset_dir, vad_mode=1):

    noisy_files_list = glob(f'{dataset_dir}/*.wav', recursive=True)

    vad = webrtcvad.Vad(vad_mode)

    total_f1 = 0
    total_fpr = 0
    total_fnr = 0
    total_precision = 0
    total_recall = 0

    for i, (file_path) in tqdm(enumerate(noisy_files_list), total=len(noisy_files_list)):

        audio_b, sr, n_frames = read_wave(file_path)
        pred_labels = np.array([vad.is_speech(frame.bytes, sr) for frame in frame_generator(10, audio_b, sr)])

        file_path = Path(file_path)
        file_id = file_path.stem
        labels_dir = str(file_path.parent)+'-labels'

        labels_file_name = os.path.join(file_path.parents[1], labels_dir, f'{file_id}.pt')

        labels = torch.load(labels_file_name).numpy()

        f1, fpr, fnr, precision, recall = get_f1(pred_labels, labels)

        total_fpr += fpr
        total_fnr += fnr
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    total = len(noisy_files_list)
    return total_fpr/total,  total_fnr/total, total_f1/total, total_precision/total, total_recall/total

if __name__ == "__main__":
    dataset_dir = '/media/administrator/Data/train-clean-100/val/LibriSpeech/dev-noisy'
    fpr, fnr, f1, precision, recall = baseline_metrics(dataset_dir)

    print(f'Baseline vad FPR = {fpr}, FNR = {fnr}, f1 = {f1}, precision = {precision}, recall = {recall}')

