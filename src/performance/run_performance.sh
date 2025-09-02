# python cot.py \
#   --n 20 --digits 2 --kinds add sub mul \
#   --backend dummy --model dummy \
#   --outdir out_dummy

pixi run python cot.py \
  --backend hf \
  --model google/gemma-2-9b-it \
  --hf_dtype auto \
  --hf_device_map auto \
  --n 200 --digits 2 4 --kinds add sub mul \
  --outdir out_hf