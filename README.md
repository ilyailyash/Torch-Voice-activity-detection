# Torch Voice activity detection

This project has been created as a test task for employment.

## Clone

Firstly, clone this repository:

```shell
git clone https://github.com/ilyailyash/Torch-Voice-activity-detection.git
cd Torch-Voice-activity-detection
```

## Environment && Installation

Install Anaconda or Miniconda, and then configure the environment with requirements.txt:

```shell
# create a conda environment
conda create --name <env> --file requirements.txt
conda activate <env>
```

## Documentation

- [Model Description](docs/model_description.md)
- [Getting Started](docs/getting_started.md)


## Pretrain models:

| Model     | EER  | FPR with FNR = 1% | FNR with FPR = 1% |
|-----------|------|-------------------|-------------------|
| [Model for `config/train/vad.toml`](https://disk.yandex.ru/d/Z0wLhgbPiSe8kg) | 9.9% | 82%               | 17%               |
| [Model for `config/train/vad_360.toml`](https://disk.yandex.ru/d/1Ozworln5biaeg) | 9.4% | 77.8%             | 13.3%             |

Presented models are on training and will be updated soon.

### Comparator with baseline model

| Model                                                | F1    |
|------------------------------------------------------|-------|
| [webrtcvad](https://github.com/wiseman/py-webrtcvad) | 0.945 |
| Proposed                                             | 0.95  |

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
