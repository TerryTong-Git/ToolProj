pixi run python coupling_probe.py \
  --datasets data1.jsonl data2.csv \
  --subsample-rate 0.01 \
  --hf-model google/gemma-2-9b-it \
  --apply-max-new 24 \
  --apply-batch-size 64 \
  --probes-per-kind 32 \
  --label-conf-thresh 0.6 \
  --min-df 20 --max-features 200000
