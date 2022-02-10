# Proposed training process and  models architectures

## Inputs

I get clean audio from LibriSpeech and with probability of 0.9 disturb it with Noise and RIR from DNS dataset.
10% clean data in training set makes converges faster and more stable. 
More details `src/dataset/LibriSpeech_train.py`

Model gets MelSpectrogram of audio samples as an input with n_mel = 80, n_fft = window_size = 320 and hop_len = 160.

Due to default padding in torch.stft (center), spectrogram frame of input signal at index j looks at 10 ms frame of index j-1 and j. 

## Model Output

Model predicts scores is range [0, 1] for each frame where 0 - silent and 1 - frame with speech.

Due to padding in torch.stft last frame could be omitted.

## True labels
I used predictions of [webrtcvad](https://github.com/wiseman/py-webrtcvad) with mode 1 on clean LibriSpeech set as true labels.

While using mode 3 for [webrtcvad](https://github.com/wiseman/py-webrtcvad) madel was overfitting hardly with 0.3-0.4 RocAuc loss on validation dataset.

Probably for cleaner metrics I could use [Allingments for LibreSpeech](https://zenodo.org/record/2619474#.YgPae7pByUl) which contains time of silens, but this wasn't done due to time limitations. [Reference](https://github.com/asteroid-team/Libri_VAD)
## Loss function

Binary Cross Entropy Loss ([nn.BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)) was used as the training objective. 
Training data wasn't sorted by length for shuffle usage. To prevent overfitting on padded audio, binary masks over loss was added. Also, to fight class disbalance weights for silent class was added: set to 4 during training.

Check more details of Loss `src/model/loss.py`

## Model Architectures:

Model is CRN with causal 2D convolution encoder, LSTM and MLP. 

Convolution block consist of:
- Causal 2D convolution with parameters: kernel_size=(5, 2), stride=(2, 1), padding=(2, 1). This leads to x2 compress over frequency domain and prevent model from looking forward in time domain.
- 2D BatchNorm
- Gelu activation

MLP has 1 hidden layer with 128 neurons.

Model was strongly inspired by [DCRNN](https://github.com/liyaguang/DCRNN).

And used hints in parameters discovered in:
- [Voxseg](https://github.com/NickWilkinson37/voxseg), 
- [NAS_VAD](https://github.com/daniel03c1/nas_vad), 
- [Unsupervised Representation Learning for Speech Activity Detection in the Fearless Steps Challenge 2021](https://www.isca-speech.org/archive/pdfs/interspeech_2021/gimeno21_interspeech.pdf),
- [A Lightweight Framework for Online Voice Activity Detection in the Wild](https://www.isca-speech.org/archive/pdfs/interspeech_2021/xu21b_interspeech.pdf)

Also, good example of RT VAD is [Silero VAD](https://github.com/snakers4/silero-vad) which could be used for knowledge distillation against [webrtcvad](https://github.com/wiseman/py-webrtcvad).

Smaller model of 0.7M parameters contains 4 conv blocs with [32, 64, 128, 128] output channels and 1 LSTM layer and Looks Ahead on 1 frame which leeds to 20 < 40 mc time latency

Bigger model of 1.3M parameters contains 4 conv blocs with [32, 64, 128, 256] output channels and 2 LSTM layer and Looks Ahead on 2 frame which leeds to 30 < 40 mc time latency

During inference cumulative normalisation over input signal is used to fit requirement.

## Metrics
ROC_AUC was used as a main quality metric over model.

EER, FPR with 1% FNR and FNR with 1% FPR was calculated during validation. More details `src/utils/metrics.py`

[python-compute-eer](https://github.com/YuanGongND/python-compute-eer)

## Optimisation
Model was trained using [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) optimiser with LR=1e-4 and weight_decay=1e-3.

[ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) was used as LR Scheduler.  
    