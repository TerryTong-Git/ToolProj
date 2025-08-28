# pixi run python eval.py \
#   --out_dir ./runs/cot_probs \
#   --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#   --prompt "A shop sold 24 apples on Monday and 31 apples on Tuesday. How many apples in total? Explain step by step, then give 'Final answer:' on its own line " \
#   --num_seeds 10 --max_new_tokens 256 --temperature 0.7 --top_p 0.95

pixi run python -m kl_div_exps.runner \
  --out_dir ./runs/jointprob_demo \
  --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --representations nl,code \
  --n_per_theta 30 --num_seeds 12 \
  --include_prompt_ll \
  --max_new_tokens 192 --temperature 0.7 --top_p 0.95 --dtype float32 

# torchrun --nproc_per_node=8 -m cot_jointprob_exp.runner \
#   --distributed \
#   --out_dir ./runs/jointprob_ddp \
#   --models TinyLlama/TinyLlama-1.1B-Chat-v1.0,meta-llama/Meta-Llama-3.1-8B-Instruct \
#   --representations nl,code \
#   --n_per_theta 50 --num_seeds 16 \
#   --include_prompt_ll

# tensorboard --logdir ./runs/jointprob_demo/tb
