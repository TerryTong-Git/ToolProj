pixi run python mi_experiment.py \
    --tb_dir ../exps_performance/out_hf/tb/run_20250902_183101 \
    --out_dir ../exps_performance/out_hf_mi_buckets \
    --model google/gemma-2-9b-it \
    --bucket_kind digits --batch_size 8 --include_prompt_ll --log_tb

# pixi run python -m vis_kl \
#   --out_dir ../runs/paired_kl \
#   --use_weighted \
#   --per_input \
#   --metric jsd --exp_id a49bce82-c727-48a2-9bc3-97d47c14827a

# pixi run tensorboard --logdir ../runs/jointprob_demo/tb
