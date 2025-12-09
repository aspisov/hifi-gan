from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable, Sequence

import torchaudio

from src.datasets.base_dataset import BaseDataset


_AUDIO_EXTENSIONS: tuple[str, ...] = (".wav", ".flac", ".mp3", ".m4a")


def _load_text(txt_path: Path) -> str:
    return txt_path.read_text(encoding="utf-8").strip()


def _audio_length_seconds(audio_path: Path) -> float:
    info = torchaudio.info(str(audio_path))
    return info.num_frames / info.sample_rate


def _find_audio_for_id(root: Path, stem: str, audio_exts: Sequence[str]) -> Path | None:
    for ext in audio_exts:
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _build_index_from_pairs(
    pairs: Iterable[tuple[Path, str]],
    audio_exts: Sequence[str],
) -> list[dict]:
    index: list[dict] = []
    for txt_path, text in pairs:
        audio_path = _find_audio_for_id(txt_path.parent.parent, txt_path.stem, audio_exts)
        if audio_path is None:
            continue
        index.append(
            {
                "path": str(audio_path.resolve()),
                "text": text,
                "audio_len": _audio_length_seconds(audio_path),
            }
        )
    return index


class CustomDirDataset(BaseDataset):
    """
    Dataset for the expected homework format:

    Root/
      <utt_id>.wav
      ...
      transcriptions/<utt_id>.txt
    """

    def __init__(
        self,
        data_dir: str | Path,
        transcription_subdir: str = "transcriptions",
        audio_extensions: Sequence[str] | None = None,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        audio_extensions = tuple(audio_extensions or _AUDIO_EXTENSIONS)

        trans_dir = data_dir / transcription_subdir
        if not trans_dir.exists():
            raise FileNotFoundError(f"Transcriptions directory not found: {trans_dir}")

        text_files = sorted(trans_dir.glob("*.txt"))
        pairs = [(txt_path, _load_text(txt_path)) for txt_path in text_files]
        index = _build_index_from_pairs(pairs, audio_extensions)

        if not index:
            raise ValueError(
                f"No audio/transcription pairs were found in {data_dir}. Check file extensions and naming."
            )

        super().__init__(index, *args, **kwargs)


class CustomDirAudioDataset(BaseDataset):
    """
    Dataset for arbitrary audio folder with optional external transcriptions.
    """

    def __init__(
        self,
        audio_dir: str | Path,
        transcription_dir: str | Path | None = None,
        audio_extensions: Sequence[str] | None = None,
        missing_text_fallback: str = "",
        *args,
        **kwargs,
    ):
        audio_dir = Path(audio_dir)
        transcription_dir = Path(transcription_dir) if transcription_dir else None
        audio_extensions = tuple(audio_extensions or _AUDIO_EXTENSIONS)

        data: list[dict] = []
        for audio_path in sorted(audio_dir.iterdir()):
            if audio_path.suffix.lower() not in audio_extensions:
                continue

            transcription = missing_text_fallback
            if transcription_dir:
                txt_path = transcription_dir / f"{audio_path.stem}.txt"
                if txt_path.exists():
                    transcription = _load_text(txt_path)

            data.append(
                {
                    "path": str(audio_path.resolve()),
                    "text": transcription,
                    "audio_len": _audio_length_seconds(audio_path),
                }
            )

        if not data:
            raise ValueError(f"No supported audio files found in {audio_dir}")

        super().__init__(data, *args, **kwargs)


class TextPromptDataset(BaseDataset):
    """
    Simple dataset for direct text prompts (no paired audio).
    Can be provided via a list, a single string, a text file, or a directory of .txt files.
    """

    def __init__(
        self,
        prompts: list[str] | str | None = None,
        prompts_file: str | Path | None = None,
        prompts_dir: str | Path | None = None,
        *args,
        **kwargs,
    ):
        texts: list[str] = []

        if prompts_dir is not None:
            pdir = Path(prompts_dir)
            txt_files = sorted(pdir.glob("*.txt"))
            for fp in txt_files:
                texts.append(fp.read_text(encoding="utf-8").strip())
        elif prompts_file is not None:
            pfile = Path(prompts_file)
            if pfile.is_file():
                texts = [line.strip() for line in pfile.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                raise FileNotFoundError(f"prompts_file not found: {prompts_file}")
        elif prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            texts = [t.strip() for t in prompts if t.strip()]

        items = [
            {
                "path": f"prompt_{idx}",
                "text": text,
                "audio_len": 0.0,
            }
            for idx, text in enumerate(texts)
        ]

        if not items:
            raise ValueError("At least one non-empty prompt is required.")

        super().__init__(items, *args, **kwargs)
