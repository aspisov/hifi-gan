import torch
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F


class HiFiGanGenerator(nn.Module):
    def __init__(self, h_u: int = 512, k_u: list = [16, 16, 4, 4]):
        super().__init__()
        self._build_architecture(h_u, k_u)
        self.tanh = nn.Tanh()

    def _build_architecture(self, base_channels, upsample_kernels):
        mel_dim = 80
        final_kernel = 7
        self.conv1 = weight_norm(nn.Conv1d(mel_dim, base_channels, kernel_size=final_kernel, padding=final_kernel // 2))
        
        upsampling_blocks = self._construct_upsampling_path(base_channels, upsample_kernels)
        self.layers = nn.Sequential(*upsampling_blocks)
        
        num_upsamples = len(upsample_kernels)
        final_channels = base_channels // (2 ** num_upsamples)
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(final_channels, 1, kernel_size=final_kernel, padding=final_kernel // 2))
        )

    def _construct_upsampling_path(self, initial_ch, kernel_list):
        block_sequence = []
        for stage_idx, kernel_val in enumerate(kernel_list):
            ch_in = initial_ch >> stage_idx
            ch_out = initial_ch >> (stage_idx + 1)
            
            block_sequence.append(nn.LeakyReLU(0.1))
            block_sequence.append(
                weight_norm(nn.ConvTranspose1d(
                    ch_in, ch_out,
                    kernel_size=kernel_val,
                    stride=kernel_val // 2,
                    padding=kernel_val // 4
                ))
            )
            block_sequence.append(MRF(ch_out))
        return block_sequence

    def forward(self, spectrogram: torch.Tensor, **batch) -> dict:
        hidden = self.conv1(spectrogram)
        hidden = self.layers(hidden)
        hidden = self.conv2(hidden)
        waveform = self.tanh(hidden)
        waveform = torch.flatten(waveform, 1)
        return {"output_audio": waveform}

    def __str__(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{super().__str__()}\nAll parameters: {total}\nTrainable parameters: {trainable}"


class HiFiGanDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = MPD()
        self.msd = MSD()

    def forward(self, output_audio, audio, detach_generated=False, **batch) -> dict:
        generated = output_audio.detach() if detach_generated else output_audio
        
        real_scale_features = self.msd(audio)
        real_period_features = self.mpd(audio)
        fake_scale_features = self.msd(generated)
        fake_period_features = self.mpd(generated)
        
        return {
            "msd_gt": real_scale_features,
            "mpd_gt": real_period_features,
            "msd_pred": fake_scale_features,
            "mpd_pred": fake_period_features,
        }

    def __str__(self):
        param_count = sum(p.numel() for p in self.parameters())
        return f"{super().__str__()}\nTotal parameters: {param_count}"


class MPD(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.periods = periods
        self.discriminators = self._create_period_discriminators(len(periods))

    def _create_period_discriminators(self, num_periods):
        channel_progression = [(1, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
        
        all_discriminators = []
        for _ in range(num_periods):
            single_disc_layers = []
            
            for idx, (ch_in, ch_out) in enumerate(channel_progression):
                norm_fn = spectral_norm if idx == 0 else weight_norm
                conv_layer = norm_fn(nn.Conv2d(ch_in, ch_out, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)))
                single_disc_layers.append(nn.Sequential(conv_layer, nn.LeakyReLU(0.1)))
            
            conv_layer = weight_norm(nn.Conv2d(1024, 1024, kernel_size=(5, 1), stride=1, padding=(2, 0)))
            single_disc_layers.append(nn.Sequential(conv_layer, nn.LeakyReLU(0.1)))
            
            final_conv = nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
            single_disc_layers.append(nn.Sequential(final_conv))
            
            all_discriminators.append(nn.ModuleList(single_disc_layers))
        
        return nn.ModuleList(all_discriminators)

    def _reshape_for_period(self, audio_signal, period_value):
        time_steps = audio_signal.size(1)
        padding_needed = (period_value - time_steps % period_value) % period_value
        
        signal = F.pad(audio_signal, (0, padding_needed), mode="reflect") if padding_needed != 0 else audio_signal
        batch_size = audio_signal.size(0)
        return signal.view(batch_size, 1, -1, period_value)

    def forward(self, input_audio):
        all_features = []
        
        for period_val, disc_layers in zip(self.periods, self.discriminators):
            reshaped = self._reshape_for_period(input_audio, period_val)
            
            layer_outputs = []
            current = reshaped
            for layer_module in disc_layers:
                current = layer_module(current)
                layer_outputs.append(current)
            
            all_features.append(layer_outputs)
        
        return all_features


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        num_scales = 3
        self.discriminators = self._build_scale_discriminators(num_scales)
        self.pools = nn.ModuleList([nn.AvgPool1d(kernel_size=4, stride=2, padding=2) for _ in range(num_scales - 1)])

    def _build_scale_discriminators(self, count):
        layer_configs = [
            (1, 16, 15, 1, 7, 1),
            (16, 64, 41, 4, 20, 4),
            (64, 256, 41, 4, 20, 16),
            (256, 1024, 41, 4, 20, 64),
            (1024, 1024, 41, 4, 20, 256),
            (1024, 1024, 5, 1, 2, 1),
            (1024, 1, 3, 1, 2, 1),
        ]
        
        disc_list = []
        for _ in range(count):
            layers_for_scale = []
            
            for ch_in, ch_out, k_size, stride_val, pad_val, group_val in layer_configs:
                conv = weight_norm(nn.Conv1d(ch_in, ch_out, kernel_size=k_size, stride=stride_val, padding=pad_val, groups=group_val))
                
                if ch_out == 1:
                    layers_for_scale.append(nn.Sequential(conv))
                else:
                    layers_for_scale.append(nn.Sequential(conv, nn.LeakyReLU(0.1)))
            
            disc_list.append(nn.ModuleList(layers_for_scale))
        
        return nn.ModuleList(disc_list)

    def _apply_pooling(self, audio_tensor, scale_idx):
        if scale_idx == 0:
            return audio_tensor
        return self.pools[scale_idx - 1](audio_tensor)

    def forward(self, audio):
        collected_features = []
        
        for scale_idx, disc_module in enumerate(self.discriminators):
            scaled_input = self._apply_pooling(audio, scale_idx)
            scaled_input = scaled_input.unsqueeze(1)
            
            intermediate_activations = []
            activation = scaled_input
            for layer_block in disc_module:
                activation = layer_block(activation)
                intermediate_activations.append(activation)
            
            collected_features.append(intermediate_activations)
        
        return collected_features


class MRF(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 7, 11], dilations=[[1, 3, 5]] * 3):
        super().__init__()
        self.blocks = self._construct_residual_blocks(channels, kernel_sizes, dilations)

    def _construct_residual_blocks(self, num_channels, kernels, dilation_patterns):
        block_list = []
        
        for kernel_val, dilation_set in zip(kernels, dilation_patterns):
            branch_layers = []
            for dilation_val in dilation_set:
                conv_block = nn.Sequential(
                    nn.LeakyReLU(negative_slope=0.1),
                    weight_norm(nn.Conv1d(num_channels, num_channels, kernel_size=kernel_val, dilation=dilation_val, padding="same"))
                )
                branch_layers.append(conv_block)
            
            block_list.append(nn.Sequential(*branch_layers))
        
        return nn.ModuleList(block_list)

    def forward(self, x):
        accumulated = torch.zeros_like(x)
        
        for residual_branch in self.blocks:
            branch_output = residual_branch(x)
            accumulated = accumulated + x + branch_output
        
        num_branches = len(self.blocks)
        return accumulated / num_branches
