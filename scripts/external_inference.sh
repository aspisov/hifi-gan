HYDRA_FULL_ERROR=1 uv run python synthesize.py -cn=inference_audio \
  datasets=audio \
  ++datasets.test.audio_dir=/workspace/hifi-gan/test_data/gt_audio \
  ++datasets.test.transcription_dir=/workspace/hifi-gan/test_data/transcriptions \
  inferencer.from_pretrained=/workspace/hifi-gan/checkpoint-epoch100-50000.pth \
  inferencer.save_path=external_test \
  writer=cometml writer.run_name=external_test_v2