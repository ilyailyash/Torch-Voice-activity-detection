# Getting Started

Project architecture inspired by [FullSubNet](https://github.com/haoxiangsnr/FullSubNet)

## Data

### Clean Speech, Noise and Room Impulse Responses

The presented model was trained of LibriSpeech corpus subsets of [100 hours](https://www.openslr.org/resources/12/train-clean-100.tar.gz) and [360 hours](https://www.openslr.org/resources/12/train-clean-360.tar.gz).
LibriSpeech dataset must be converted to wav by scripts/preprocessing_dataset.py 

```shell
python scripts/preprocessing_dataset.py -D path_to_librispeech_dataset
```

The Noise and RIR data was used from DNS Challenge dataset (ICASSP 2021) [https://github.com/microsoft/DNS-Challenge.git](https://github.com/microsoft/DNS-Challenge.git).

Now noise and RIRs could be downloaded using [download-dns-challenge-3.sh](https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-3.sh)

Any datasets of .wav clean, noise or rirs could be used for training.   

### Validation set preparation

In these experiments LibriSpeech dev corpus [dev-clean.tar.gz](https://www.openslr.org/resources/12/dev-clean.tar.gz) was used as a base for validation set.

To generate validation set use scripts/syntesize_devset.py, you should manually change datasets dirs for clean, noise and rirs in .py file:

```shell
python scripts/syntesize_devset.py
```

## Usage

### Training

We could call the default training configuration, You need to change paths to data:

```shell
# enter a directory named after the dataset, such as dns_interspeech_2020
cd FullSubNet/recipes/dns_interspeech_2020

# Use a default config and two GPUs to train the small model
train.py -C ./config/train/vad.toml -W 2

# Use a default config and two GPUs to train more complex model
train.py -C ./config/train/vad_360.toml -W 2

# Resume the experiment using "-R" parameter
train.py -C ./config/train/vad.toml -W 2 -R
```

### Logs and Visualization

The logs during the training will be stored, and we can visualize it using TensorBoard. Assuming that:

- The file path of the training configuration is `config/train/vad.toml`

Then, the log information will be stored in the `~/experiments/vad` directory. This directory contains the following:

- `logs/` directory: store the TensorBoard related data, including loss curves, metrics and det curves.
- `checkpoints/` directory: stores all checkpoints during training, from which you can resume the training or start an inference.
- `*.toml` file: the backup of the current training configuration.

 In the `logs/` directory, use the following command to visualize loss curves during the training and the validation.

```shell
tensorboard --logdir ~/experiments/vad

# specify a port 45454
tensorboard --logdir ~/experiments/vad --port 45454
```

### Inference

After training, you can save vad masks and evaluate metrics on speech. Take the vad as an example:

1. Checking the noisy speech directory path and the sample rate in `config/common/inference.toml`.

```toml
[dataset.args]
dataset_dir = "/path/to/your/dataset/"
```

2. Switch to project root directory and start inference:

```shell

# One GPU is used by default
python inference.py \
  -C ./config/inference/vad.toml \
  -M /path/to/your/checkpoint_dir/best_model.tar \
  -O /path/to/your/results_dir
```
Results_dir will contain configuration file .toml and folder with:
 - .txt file with scores on audio in inference datafolder 
 - .txt file with trained thresholds for eer and 1% FPR and FNR

### Applying a Pre-trained Model

In the inference stage, you can use a pre-trained model given in -M argument:

Check more details of inference parameters in `config/inference/vad_inference.toml`.

### Metrics

Calculating metrics (ROC_AUC, EER, FPR with FNR=1%, FNR with FPR=1%)

