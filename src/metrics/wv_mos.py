from pathlib import Path
import tempfile

import torch
import torchaudio
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.serialization as serialization
from wvmos import get_wvmos

from src.metrics.base_metric import BaseMetric


class WV_MOS_Metric(BaseMetric):
    def __init__(
        self,
        name: str | None = None,
        source_sr: int = 22050,
        target_sr: int = 16000,
        device: str = "auto",
        audio_dir: str | None = None,
    ):
        super().__init__(name=name)
        device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        serialization.add_safe_globals([ModelCheckpoint])
        self.model = get_wvmos(cuda=device == "cuda")
        self.source_sr = source_sr
        self.target_sr = target_sr
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None

    def __call__(
        self,
        output_audio: torch.Tensor | None = None,
        audio_dir: str | None = None,
        **kwargs,
    ) -> float:
        directory = audio_dir or self.audio_dir
        if directory is not None:
            return float(self.model.calculate_dir(str(directory), mean=True))

        audio_batch = output_audio.detach().cpu()
        if audio_batch.ndim == 2:
            audio_batch = audio_batch.unsqueeze(1)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            for idx, audio in enumerate(audio_batch):
                audio = audio.to(torch.float32)
                if self.source_sr != self.target_sr:
                    audio = torchaudio.functional.resample(audio, self.source_sr, self.target_sr)
                torchaudio.save(
                    str(tmpdir_path / f"sample_{idx}.wav"),
                    audio,
                    sample_rate=self.target_sr,
                )
            score = self.model.calculate_dir(str(tmpdir_path), mean=True)
        return float(score)
