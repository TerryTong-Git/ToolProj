pixi run python semantics.py \
  --tb_dir ../exps_performance/out_hf_scale/tb/run_20250904_113115 \
  --out_dir ./out_hf_scale_mi_arith \
  --model google/gemma-2-9b-it \
  --engine vllm \
  --tensor_parallel_size 8 \
  --gpu_memory_util 0.5 \
  --vllm_dtype float16 \
  --max_model_len 2048 \
  --bucket_kind digits \
  --batch_size 128 \
  --score_batch_size 1 \
  --apply_max_new_tokens 128 \
  --digits 2,4,8,16,32 \
  --kinds add,sub,mul \
  --log_tb --probes_total_per_bucket 64 --probes_per_kind 8 \
  --log_level INFO --limit_semantics 200

