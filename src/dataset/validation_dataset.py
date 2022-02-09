import os
from pathlib import Path

import torch
import torchaudio
from glob import glob

from src.common.dataset import BaseDataset
from src.util.acoustic_utils import load_wav


class Dataset(BaseDataset):
    def __init__(
            self,
            dataset_dir,
            sr,
    ):
        """
        Construct DNS validation set

        LibriSpeech/
            dev-noisy/
            dev-noisy-labels/
        """
        super(Dataset, self).__init__()
        noisy_files_list = glob(f'{dataset_dir}/*.wav', recursive=True)

        self.length = len(noisy_files_list)
        self.noisy_files_list = noisy_files_list
        self.sr = sr

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        """
        use the absolute path of noisy speeches to find the corresponding labels.

        Returns:
            [waveform...], [labels...]
        """
        noisy_file_path = self.noisy_files_list[item]
        noisy_file_path = Path(noisy_file_path)
        file_id = noisy_file_path.stem
        labels_dir = str(noisy_file_path.parent)+'-labels'

        labels_file_name = os.path.join(noisy_file_path.parents[1], labels_dir, f'{file_id}.pt')

        noisy, _ = torchaudio.load(noisy_file_path)
        labels = torch.load(labels_file_name)

        return noisy[0].float(), labels.float()
