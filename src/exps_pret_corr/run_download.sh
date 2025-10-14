export HF_TOKEN=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
pixi run python download_hf_manual.py \
  --hf_dataset_id mlfoundations/dclm-baseline-1.0 \
  --revision main \
  --gcs_output_path ./data/dclm \
  --max_files 100

pixi run python download_hf_manual.py \
  --hf_dataset_id bigcode/starcoderdata \
  --revision main \
  --gcs_output_path ./data/starcoder \
  --max_files 100



# python -m cProfile script.py