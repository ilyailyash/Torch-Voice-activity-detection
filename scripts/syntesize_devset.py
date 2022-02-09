import os

from tqdm import tqdm
import torch
import torchaudio
from torch.utils.data import DataLoader

from src.dataset.LibriSpeech_train import Dataset


def generate_set(dataset, base_path, noisy_dir, label_dir, n=1):
    '''
    Saves labels and audio files in base_path folder

    :param dataset: torch dataset that creates noisy audio
    :param base_path: path to new data dir
    :param noisy_dir: folder name of noisy files
    :param label_dir: folder name of labels
    :param n: how many times generate noisy files from full dataset in dataloader

    :return:
    '''

    dataloader = DataLoader(dataset=dataset, batch_size=1)
    i = 0
    # synthesize and save set for validation
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, noisy_dir), exist_ok=True)
    os.makedirs(os.path.join(base_path, label_dir), exist_ok=True)

    for j in range(n):
        for noisy, labels in tqdm(dataloader, total=len(dataloader)):
            wav_path = os.path.join(base_path, noisy_dir, f'{i}.wav')
            torchaudio.save(wav_path, noisy, sr, encoding="PCM_S", bits_per_sample=16)

            labels_path = os.path.join(base_path, label_dir, f'{i}.pt')
            torch.save(labels[0], labels_path)
            i += 1


if __name__ == "__main__":
    clean_dataset = "/media/administrator/Data/train-clean-100/val/LibriSpeech/dev-clean_wav"
    # clean_dataset = "/media/administrator/Data/train-clean-100/LibriSpeech/train-clean-100_wav"
    clean_dataset_limit = False
    clean_dataset_offset = 0
    noise_dataset = "/media/administrator/Data/DNS-Challenge/datasets/noise"
    noise_dataset_limit = False
    noise_dataset_offset = 0
    rir_dataset = "/media/administrator/Data/DNS-Challenge/datasets/impulse_responses"
    rir_dataset_limit = False
    rir_dataset_offset = 0
    snr_range = [-5, 20]
    reverb_proportion = 0.5
    silence_length = 0.2
    target_dB_FS = -25
    target_dB_FS_floating_value = 10
    sr = 16000
    pre_load_clean_dataset = False
    pre_load_noise = False
    pre_load_rir = False
    num_workers = 36
    vad_mode = 1
    data_bit = 16
    noise_proportion = 1

    dataset = Dataset(clean_dataset,
                      clean_dataset_limit,
                      clean_dataset_offset,
                      noise_dataset,
                      noise_dataset_limit,
                      noise_dataset_offset,
                      rir_dataset,
                      rir_dataset_limit,
                      rir_dataset_offset,
                      snr_range,
                      reverb_proportion,
                      silence_length,
                      target_dB_FS,
                      target_dB_FS_floating_value,
                      sr,
                      pre_load_clean_dataset,
                      pre_load_noise,
                      pre_load_rir,
                      num_workers,
                      vad_mode,
                      data_bit,
                      noise_proportion
                      )


    #valid
    base_path = "/media/administrator/Data/train-clean-100/val/LibriSpeech"
    generate_set(dataset, base_path, 'dev-noisy', 'dev-noisy-labels')

    #train
    # base_path = "/media/administrator/Data/train-clean-100/train/LibriSpeech"
    # generate_set(dataset, base_path, 'train-clean', 'train-clean-labels')