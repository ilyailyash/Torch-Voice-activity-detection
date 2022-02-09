# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import pathlib

import torch
import torchaudio

from glob import glob
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        self.filenames = glob(f'{path}/**/*.wav', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        audio, sr = torchaudio.load(filename)
        if sr != 16000:
            raise ValueError('sample rate in not 16000')
        return audio[0], pathlib.Path(filename).name


def from_path(noisy_path):
    dataset = Dataset(noisy_path)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=1,
                                       shuffle=False,
                                       pin_memory=False)
