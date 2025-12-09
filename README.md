# HiFi-GAN

## Report

[Comet ML report](https://www.comet.com/dmitriy-aspisov/hifi-gan/reports/FbfbMJKUTULlWbC1rF1ZQpO5A)

## Train

To train model with base config run:
```sh 
sh scripts/train.sh
```

## Inference

First you need to download best model:
```sh
sh scripts/download_model.sh
```

You also need to download custom dataset:
```sh
sh scripts/download_dataset.sh
```

To run inference from audio:
```sh 
sh scripts/external_inference.sh
```

To run inference for full TTS system run:
```sh 
sh scripts/full_tts_inference.sh
```

## Demo notebook
You should run this demo notebook from [google colab](https://colab.research.google.com/drive/1R1WJVxK6ldhveK1ChIVDXUUrDsTLY1el?usp=sharing)
