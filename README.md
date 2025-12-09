# HiFi-GAN

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