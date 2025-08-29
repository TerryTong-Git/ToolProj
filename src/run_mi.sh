pixi run python -m mi_exps.mi_experiment \
  --out_dir ./runs/paired_kl \
  --theta_prior weighted \
  --alpha 1e-3 \
  --use_weighted \
  --log_tb --exp_id a49bce82-c727-48a2-9bc3-97d47c14827a

pixi run python -m mi_exps.vis_kl \
  --out_dir ./runs/paired_kl \
  --use_weighted \
  --per_input \
  --metric jsd --exp_id a49bce82-c727-48a2-9bc3-97d47c14827a
