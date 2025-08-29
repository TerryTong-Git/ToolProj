pixi run python -m kl_div_exps.runner \
  --out_dir ./runs/jointprob_demo \
  --models google/gemma-2-9b-it \
  --representations nl,code \
  --n_per_theta 8 --num_seeds 5 \
  --include_prompt_ll \
  --max_new_tokens 256 --temperature 0.7 --top_p 0.95 --dtype float32