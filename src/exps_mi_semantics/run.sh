pixi run python semantics.py \
  --tb_dir ../exps_performance/out_hf_scale_deepseek_working/tb/run_20250909_105603 \
  --out_dir ./deepseek \
  --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
  --engine vllm \
  --tensor_parallel_size 8 \
  --gpu_memory_util 0.5 \
  --vllm_dtype float16 \
  --max_model_len 2048 \
  --bucket_kind all \
  --batch_size 128 \
  --score_batch_size 1 \
  --apply_max_new_tokens 128 \
  --digits 2,4,8,16,32 \
  --kinds add,sub,mul,lcs,knap,rod,ilp_prod,ilp_assign,ilp_partition \
  --log_tb --probes_total_per_bucket 64 --probes_per_kind 8 \
  --log_level INFO --limit_semantics 100

pixi run python semantics.py \
  --tb_dir ../exps_performance/out_hf_scale_deepseek_working/tb/run_20250909_113629 \
  --out_dir ./deepseek1 \
  --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
  --engine vllm \
  --tensor_parallel_size 8 \
  --gpu_memory_util 0.5 \
  --vllm_dtype float16 \
  --max_model_len 2048 \
  --bucket_kind all \
  --batch_size 128 \
  --score_batch_size 1 \
  --apply_max_new_tokens 128 \
  --digits 2,4,8,16,32 \
  --kinds add,sub,mul,lcs,knap,rod,ilp_prod,ilp_assign,ilp_partition \
  --log_tb --probes_total_per_bucket 64 --probes_per_kind 8 \
  --log_level INFO --limit_semantics 100

pixi run python semantics.py \
  --tb_dir ../exps_performance/out_hf_scale_deepseek_working/tb/run_20250909_121543 \
  --out_dir ./deepseek2 \
  --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
  --engine vllm \
  --tensor_parallel_size 8 \
  --gpu_memory_util 0.5 \
  --vllm_dtype float16 \
  --max_model_len 2048 \
  --bucket_kind all \
  --batch_size 128 \
  --score_batch_size 1 \
  --apply_max_new_tokens 128 \
  --digits 2,4,8,16,32 \
  --kinds add,sub,mul,lcs,knap,rod,ilp_prod,ilp_assign,ilp_partition \
  --log_tb --probes_total_per_bucket 64 --probes_per_kind 8 \
  --log_level INFO --limit_semantics 100

pixi run python semantics.py \
  --tb_dir ../exps_performance/out_hf_scale_deepseek_working/tb/run_20250909_125525 \
  --out_dir ./deepseek3 \
  --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
  --engine vllm \
  --tensor_parallel_size 8 \
  --gpu_memory_util 0.5 \
  --vllm_dtype float16 \
  --max_model_len 2048 \
  --bucket_kind all \
  --batch_size 128 \
  --score_batch_size 1 \
  --apply_max_new_tokens 128 \
  --digits 2,4,8,16,32 \
  --kinds add,sub,mul,lcs,knap,rod,ilp_prod,ilp_assign,ilp_partition \
  --log_tb --probes_total_per_bucket 100 --probes_per_kind 8 \
  --log_level INFO --limit_semantics 100

pixi run python semantics.py \
  --tb_dir ../exps_performance/out_hf_scale_deepseek_working/tb/run_20250909_133339 \
  --out_dir ./deepseek4 \
  --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
  --engine vllm \
  --tensor_parallel_size 8 \
  --gpu_memory_util 0.5 \
  --vllm_dtype float16 \
  --max_model_len 2048 \
  --bucket_kind all \
  --batch_size 128 \
  --score_batch_size 1 \
  --apply_max_new_tokens 128 \
  --digits 2,4,8,16,32 \
  --kinds add,sub,mul,lcs,knap,rod,ilp_prod,ilp_assign,ilp_partition \
  --log_tb --probes_total_per_bucket 64 --probes_per_kind 8 \
  --log_level INFO --limit_semantics 100
