

seeds=(0)

for seed in ${seeds[@]}; do
  pixi run python cot.py \
  --backend openai \
  --model gpt-5-nano \
  --n 100 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
  --outdir openai --exec_code --batch_size 8
done





  




