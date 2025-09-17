pixi run python mi_experiment.py \
    --tb_dir ../exps_performance/out_hf_scale_deepseek_working/tb/run_20250909_105603 \
    --out_dir ./deepseek_digits \
    --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
    --bucket_kind digits --batch_size 1 --log_tb --engine vllm --vllm_dtype float16 --seed 0

pixi run python mi_experiment.py \
    --tb_dir ../exps_performance/out_hf_scale_deepseek_working/tb/run_20250909_113629 \
    --out_dir ./deepseek1_digits \
    --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
    --bucket_kind digits --batch_size 1 --log_tb --engine vllm --vllm_dtype float16 --seed 0

pixi run python mi_experiment.py \
    --tb_dir ../exps_performance/out_hf_scale_deepseek_working/tb/run_20250909_121543 \
    --out_dir ./deepseek2_digits \
    --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
    --bucket_kind digits --batch_size 1 --log_tb --engine vllm --vllm_dtype float16 --seed 0

pixi run python mi_experiment.py \
    --tb_dir ../exps_performance/out_hf_scale_deepseek_working/tb/run_20250909_125525 \
    --out_dir ./deepseek3_digits \
    --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
    --bucket_kind digits --batch_size 1 --log_tb --engine vllm --vllm_dtype float16 --seed 0

pixi run python mi_experiment.py \
    --tb_dir ../exps_performance/out_hf_scale_deepseek_working/tb/run_20250909_133339 \
    --out_dir ./deepseek4_digits \
    --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
    --bucket_kind digits --batch_size 1 --log_tb --engine vllm --vllm_dtype float16 --seed 0

