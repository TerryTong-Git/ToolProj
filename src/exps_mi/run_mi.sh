pixi run python mi_experiment.py \
  --out_dir ../out_hf \
  --model google/gemma-2-9b-it \
  --include_prompt_ll \
  --use_weighted \
  --mi_tb

# pixi run python -m vis_kl \
#   --out_dir ../runs/paired_kl \
#   --use_weighted \
#   --per_input \
#   --metric jsd --exp_id a49bce82-c727-48a2-9bc3-97d47c14827a

# pixi run tensorboard --logdir ../runs/jointprob_demo/tb
