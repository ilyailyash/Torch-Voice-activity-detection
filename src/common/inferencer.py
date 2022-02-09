import time
from functools import partial
from pathlib import Path
from collections import OrderedDict

import librosa
import toml
import torch
from torch.utils.data import DataLoader
from src.util.acoustic_utils import stft, istft
from src.util.utils import initialize_module, prepare_device, prepare_empty_dir


class BaseInferencer:
    def __init__(self, config, checkpoint_path, output_dir):
        checkpoint_path = Path(checkpoint_path).expanduser().absolute()
        root_dir = Path(output_dir).expanduser().absolute()
        self.device = prepare_device(torch.cuda.device_count())
        # self.device = torch.device("cpu")


        print("Loading inference dataset...")
        self.dataloader = self._load_dataloader(config["dataset"])
        print("Loading model...")

        self.model, epoch, thresholds = self._load_model(config["model"], checkpoint_path, self.device)
        self.thresholds = thresholds
        self.inference_config = config["inferencer"]

        self.scores_dir = root_dir / f"vad_{str(epoch).zfill(4)}"

        prepare_empty_dir([self.scores_dir])

        self.acoustic_config = config["acoustic"]
        n_fft = self.acoustic_config["n_fft"]
        hop_length = self.acoustic_config["hop_length"]
        win_length = self.acoustic_config["win_length"]

        self.stft = partial(stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length, device=self.device)
        self.istft = partial(istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length, device=self.device)
        self.librosa_stft = partial(librosa.stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.librosa_istft = partial(librosa.istft, hop_length=hop_length, win_length=win_length)

        print("Configurations are as follows: ")
        print(toml.dumps(config))
        with open((root_dir / f"{time.strftime('%Y-%m-%d %H:%M:%S')}.toml").as_posix(), "w") as handle:
            toml.dump(config, handle)

    @staticmethod
    def _load_dataloader(dataset_config):
        dataset = initialize_module(dataset_config["path"], args=dataset_config["args"], initialize=True)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
        )
        return dataloader

    @staticmethod
    def _load_model(model_config, checkpoint_path, device):

        model = initialize_module(model_config["path"], args=model_config["args"])

        model_checkpoint = torch.load(checkpoint_path, map_location=device)
        model_static_dict = model_checkpoint["model"]
        epoch = model_checkpoint["epoch"]
        print(f"The model breakpoint in tar format is currently being processed, and its epoch is：{epoch}.")

        new_state_dict = OrderedDict()
        for k, v in model_static_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
            
        # load params
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        return model, model_checkpoint["epoch"], model_checkpoint["thresholds"]

    def inference(self):
        raise NotImplementedError
