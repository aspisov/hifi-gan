from itertools import chain
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    audios = list(chain.from_iterable(item["audio"] if item["audio"] is not None else [] for item in dataset_items))
    audios = pad_sequence(audios, batch_first=True) if audios else audios

    spectrograms = list(chain.from_iterable(item["spectrogram"].transpose(1, 2) for item in dataset_items))
    spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=-11.5129251).transpose(1, 2)

    texts = [item["text"] for item in dataset_items]

    return {"audio": audios, "spectrogram": spectrograms, "text": texts}
