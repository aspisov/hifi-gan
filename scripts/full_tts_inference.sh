uv run python synthesize.py -cn=inference_text \
  ++datasets.test.prompts_dir=/workspace/hifi-gan/data/datasets/ljspeech/transcriptions \
  ++inferencer.gt_audio_dir=/workspace/hifi-gan/data/datasets/ljspeech/test \
  inferencer.from_pretrained=/workspace/hifi-gan/checkpoint-epoch100-50000.pth \
  inferencer.save_path=full_tts_lj_text \
  writer=cometml writer.run_name=full_tts_lj_text_v2

uv run python synthesize.py -cn=inference_text \
  ++datasets.test.prompts_dir=/workspace/hifi-gan/test_data/transcriptions \
  ++inferencer.gt_audio_dir=/workspace/hifi-gan/test_data/gt_audio \
  inferencer.from_pretrained=/workspace/hifi-gan/checkpoint-epoch100-50000.pth \
  inferencer.save_path=full_tts_ext_text \
  writer=cometml writer.run_name=full_tts_ext_text_v2