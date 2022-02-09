import warnings

import torch
from torch.cuda.amp import autocast
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
import matplotlib.pyplot as plt
from prefetch_generator import BackgroundGenerator

from src.common.trainer import BaseTrainer

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            dist,
            rank,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            scheduler,
            train_dataloader,
            validation_dataloader
    ):
        super(Trainer, self).__init__(dist,
                                      rank,
                                      config,
                                      resume,
                                      model,
                                      loss_function,
                                      optimizer,
                                      scheduler)

        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

        n_fft = self.acoustic_config["n_fft"]
        hop_length = self.acoustic_config["hop_length"]
        win_length = self.acoustic_config["win_length"]
        center = self.acoustic_config["center"]
        n_mel = self.acoustic_config["n_mel"]

        self.mel_spectrogram = MelSpectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center,
                                              n_mels=n_mel)

    def apply_mask(self, audio, labels, delete_clean=False):
        assert audio.size(0) == 1, f'batch size should be equal to 1, but {audio.size(0)}'
        # [1, n_fft, n_frames, 2]
        audio_stft = self.torch_stft(audio)
        # [1, n_fft, n_frames, 2]
        audio_stft = audio_stft.permute(0, 1, 3, 2)

        if labels.shape[-1] != audio_stft.shape[-1]:
            warnings.warn('n frames are not the same in applying')

        if delete_clean:
            labels = (labels == 0).float()

        audio_stft[:, :, :, :labels.shape[-1]] = audio_stft[:, :, :, :labels.shape[-1]]*labels
        # [1, n_fft, n_frames, 2]
        audio_stft = audio_stft.permute(0, 1, 3, 2)
        audio = self.istft(audio_stft)

        return audio

    def _train_epoch(self, epoch):
        loss_total = 0.0

        i = 0
        desc = f"Training {self.rank}"
        with tqdm(self.train_dataloader, desc=desc, total=len(self.train_dataloader)) as pgbr:
            for noisy, labels, mask in pgbr:
                self.optimizer.zero_grad()

                noisy = noisy.to(self.rank)
                labels = labels.to(self.rank)
                mask = mask.to(self.rank)

                self.mel_spectrogram = self.mel_spectrogram.to(self.rank)

                noisy_amp = self.mel_spectrogram(noisy)

                with autocast(enabled=self.use_amp):
                    # [B, 1, F, T] => model => [B, T]
                    pred_scores = self.model(noisy_amp.unsqueeze(1))
                    pred_scores = pred_scores[:, :labels.size(-1)]

                    loss = self.loss_function(pred_scores, labels, mask)

                # audio = self.apply_mask(noisy.to(self.rank)[0:1], labels[0:1], True)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # if self.rank == 0 and i % 20 == 0:
                #     self.writer.add_scalar(f"Loss/loss", loss.item(), i)
                #     self.writer.add_scalar(f"Loss/LR", self.optimizer.param_groups[0]['lr'], i)
                # i += 1
                #
                # self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*1.01

                loss_total += loss.item()
                pgbr.desc = desc + ' loss = {:5.3f}'.format(loss.item())

            if self.rank == 0:
                self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualization_metrics = self.visualization_config["metrics"]

        loss_total = 0.0
        pred_scores_list = []
        labels_list = []

        for i, (noisy, labels) in tqdm(enumerate(self.valid_dataloader),
                                       total=len(self.valid_dataloader),
                                       desc="Validation"):
            assert noisy.shape[0] == 1, "The batch size of validation stage must be one."

            noisy_amp = self.mel_spectrogram(noisy).to(self.rank)

            labels = labels.to(self.rank)

            pred_scores = self.model(noisy_amp.unsqueeze(1))
            pred_scores = pred_scores[:, :labels.size(-1)]

            loss = self.loss_function(pred_scores, labels)

            loss_total += loss

            pred_scores_list.append(pred_scores.to('cpu'))
            labels_list.append(labels.to('cpu'))

        self.writer.add_scalar(f"Loss/Validation_Total", loss_total / len(self.valid_dataloader), epoch)

        validation_score = self.metrics_visualization(
            torch.cat(labels_list, 1), torch.cat(pred_scores_list, 1), visualization_metrics, epoch,
        )

        self.scheduler.step(validation_score)

        del pred_scores_list
        del labels_list

        return validation_score
