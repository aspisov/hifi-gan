import torch
from torch import nn
import torch.nn.functional as F

from src.transforms import MelSpectrogram, MelSpectrogramConfig


def _adv_mse(pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return torch.mean((pred - target) ** 2)


class PeriodDiscriminatorLoss(nn.Module):
    def forward(self, real_outputs, fake_outputs):
        loss = 0.0
        for real_disc, fake_disc in zip(real_outputs, fake_outputs):
            loss += _adv_mse(real_disc[-1], True)
            loss += _adv_mse(fake_disc[-1], False)
        return loss


class ScaleDiscriminatorLoss(nn.Module):
    def forward(self, real_outputs, fake_outputs):
        loss = 0.0
        for real_disc, fake_disc in zip(real_outputs, fake_outputs):
            loss += _adv_mse(real_disc[-1], True)
            loss += _adv_mse(fake_disc[-1], False)
        return loss


class FeatureMatching(nn.Module):
    def forward(self, real_outputs, fake_outputs):
        fm_loss = 0.0
        for real_disc, fake_disc in zip(real_outputs, fake_outputs):
            for real_feat, fake_feat in zip(real_disc[:-1], fake_disc[:-1]):
                fm_loss += F.l1_loss(fake_feat, real_feat)
        return fm_loss


class HiFiGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.period_loss = PeriodDiscriminatorLoss()
        self.scale_loss = ScaleDiscriminatorLoss()

    def forward(self, msd_gt, mpd_gt, msd_pred, mpd_pred, **kwargs):
        d_mpd_loss = self.period_loss(mpd_gt, mpd_pred)
        d_msd_loss = self.scale_loss(msd_gt, msd_pred)
        d_loss = d_mpd_loss + d_msd_loss
        return {
            "d_loss": d_loss,
            "d_mpd_loss": d_mpd_loss,
            "d_msd_loss": d_msd_loss,
        }


class HiFiGANGeneratorLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.mel_spectrogram = MelSpectrogram(MelSpectrogramConfig()).to(self.device)
        self.feature_matching = FeatureMatching()

    def forward(self, msd_gt, mpd_gt, msd_pred, mpd_pred, output_audio, spectrogram, **batch):
        spectrogram = spectrogram.to(self.device)
        mel_output = self.mel_spectrogram(output_audio)

        min_time = min(mel_output.shape[-1], spectrogram.shape[-1])
        mel_output = mel_output[..., :min_time]
        spectrogram = spectrogram[..., :min_time]
        mel_loss = F.l1_loss(mel_output, spectrogram)

        adv_loss = 0.0
        for disc_out in msd_pred:
            adv_loss += _adv_mse(disc_out[-1], True)
        for disc_out in mpd_pred:
            adv_loss += _adv_mse(disc_out[-1], True)

        feature_loss = self.feature_matching(msd_gt, msd_pred) + self.feature_matching(mpd_gt, mpd_pred)

        total = adv_loss + 2 * feature_loss + 45 * mel_loss
        return {
            "g_loss": total,
            "g_adv_loss": adv_loss,
            "g_feat_loss": feature_loss,
            "g_mel_loss": mel_loss,
        }
