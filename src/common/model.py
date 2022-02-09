import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def sband_forgetting_norm(input, train_sample_length):
        """
        与 forgetting norm
        Args:
            input:
            train_sample_length:

        Returns:

        """
        assert input.ndim == 3
        batch_size, n_freqs, n_frames = input.size()

        eps = 1e-10
        alpha = (train_sample_length - 1) / (train_sample_length + 1)
        mu = 0
        mu_list = []

        for idx in range(input.shape[-1]):
            if idx < train_sample_length:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
            else:
                mu = alpha * mu + (1 - alpha) * input[:, (n_freqs // 2 - 1), idx].reshape(batch_size, 1)

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        input = input / (mu + eps)
        return input

    @staticmethod
    def forgetting_norm(input, sample_length_in_training):
        """
        输入为三维，通过不断估计邻近的均值来作为当前 norm 时的均值

        Args:
            input: [B, F, T]
            sample_length_in_training: 训练时的长度，用于计算平滑因子

        Returns:

        """
        assert input.ndim == 3
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10
        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)

        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
            else:
                current_frame_mu = torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu = alpha * mu + (1 - alpha) * current_frame_mu

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        input = input / (mu + eps)
        return input

    @staticmethod
    def hybrid_norm(input, sample_length_in_training=192):
        assert input.ndim == 3
        device = input.device
        data_type = input.dtype
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10

        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)
        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu_list.append(mu)
            else:
                break
        initial_mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]

        step_sum = torch.sum(input, dim=1)  # [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
        entry_count = entry_count.reshape(1, n_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cum_mean = cumulative_sum / entry_count  # B, T

        cum_mean = cum_mean.reshape(batch_size, 1, n_frames)  # [B, 1, T]

        # print(initial_mu[0, 0, :50])
        # print("-"*60)
        # print(cum_mean[0, 0, :50])
        cum_mean[:, :, :sample_length_in_training] = initial_mu

        return input / (cum_mean + eps)

    @staticmethod
    def cumulative_norm(input):
        eps = 1e-10
        device = input.device
        data_type = input.dtype
        n_dim = input.ndim

        assert n_dim in (3, 4)

        if n_dim == 3:
            n_channels = 1
            batch_size, n_freqs, n_frames = input.size()
        else:
            batch_size, n_channels, n_freqs, n_frames = input.size()
            input = input.reshape(batch_size * n_channels, n_freqs, n_frames)

        step_sum = torch.sum(input, dim=1)  # [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
        entry_count = entry_count.reshape(1, n_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cum_mean = cumulative_sum / entry_count  # B, T

        cum_mean = cum_mean.reshape(batch_size * n_channels, 1, n_frames)

        x = input / (cum_mean + eps)

        if n_dim == 4:
            x = x.reshape(batch_size, n_channels, n_freqs, n_frames)

        return x

    @staticmethod
    def cumulative_layer_norm(input):
        eps = 1e-10
        device = input.device
        data_type = input.dtype
        n_dim = input.ndim

        assert n_dim in (3, 4)

        if n_dim == 3:
            n_channels = 1
            batch_size, n_freqs, n_frames = input.size()
        else:
            batch_size, n_channels, n_freqs, n_frames = input.size()
            input = input.reshape(batch_size * n_channels, n_freqs, n_frames)

        step_sum = torch.sum(input, dim=1)  # [B, T]
        step_pow_sum = torch.sum(torch.square(input), dim=1)

        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
        cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
        entry_count = entry_count.reshape(1, n_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cum_mean = cumulative_sum / entry_count  # B, T
        cum_var = (cumulative_pow_sum - 2 * cum_mean * cumulative_sum) / entry_count + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + eps).sqrt()  # B, T

        cum_mean = cum_mean.reshape(batch_size * n_channels, 1, n_frames)
        cum_std = cum_std.reshape(batch_size * n_channels, 1, n_frames)

        x = (input - cum_mean) / cum_std

        if n_dim == 4:
            x = x.reshape(batch_size, n_channels, n_freqs, n_frames)

        return x

    def weight_init(self, m):
        """
        Usage:
            model = Model()
            model.apply(weight_init)
        """
        if isinstance(m, nn.Conv1d):
            init.xavier_uniform_(m.weight.data, gain=init.calculate_gain('relu'))
            if m.bias is not None:
                init.zeros_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data, gain=init.calculate_gain('relu'))
            if m.bias is not None:
                init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.zeros_(m.bias.data)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data)
            init.zeros_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.xavier_uniform_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.xavier_uniform_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.xavier_uniform_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.xavier_uniform_(param.data)
