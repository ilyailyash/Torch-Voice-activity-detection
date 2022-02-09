import os
import argparse
from pathlib import Path

from tqdm import tqdm
from glob import glob
from pydub import AudioSegment


def flac_to_wav_librispeech(data_dir):
    _ext_audio = ".wav"
    walker = glob(f'{data_dir}/**/*.flac', recursive=True)

    save_path = data_dir + '_wav'

    for file_name in tqdm(walker):
        fileid = Path(file_name).stem

        speaker_id, chapter_id, utterance_id = fileid.split("-")
        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id

        flac = AudioSegment.from_file(file_name, format='flac')

        save_file_audio_name = fileid_audio + '.wav'
        save_file_audio = os.path.join(save_path, speaker_id, chapter_id)

        os.makedirs(save_file_audio, exist_ok=True)

        save_file_audio = os.path.join(save_file_audio, save_file_audio_name)

        flac.export(save_file_audio, format='wav')


if __name__ == '__main__':
    """
    converts librispeech from .flac to .wav and saves it
    """
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("-D", "--path_to_librispeech_dataset", type=str, required=True, help="Configuration file.")
    args = parser.parse_args()

    data_dir = args.path_to_librispeech_dataset

    # data_dir = '/media/administrator/Data/LibriSpeech/train-clean-360'
    flac_to_wav_librispeech(data_dir)
