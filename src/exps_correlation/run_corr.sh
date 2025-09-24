seed=(0 1 2 3 4)
for seed in ${seeds[@]}; do
    pixi run python run_corr.py --models meta-llama/Llama-3.1-8B-Instruct --engine vllm --temperature 0.3 --top_p 0.95 --seed ${seed}
    pixi run python run_corr.py --models deepseek-ai/DeepSeek-R1-Distill-Llama-8B --engine vllm --temperature 0.3 --top_p 0.95 --seed ${seed}
    pixi run python run_corr.py --models google/gemma-2-9b-it --engine vllm --temperature 0.3 --top_p 0.95 --seed ${seed}
    pixi run python run_corr.py --models deepseek-ai/deepseek-coder-7b-instruct-v1.5  --engine vllm --temperature 0.3 --top_p 0.95 --seed ${seed}
    pixi run python run_corr.py --models mistralai/Ministral-8B-Instruct-2410 --engine vllm --temperature 0.3 --top_p 0.95 --seed ${seed}
done