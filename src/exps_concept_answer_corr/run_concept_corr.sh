# python concept_answer_correlation.py \
#   --samples-per-concept 10 \
#   --engine vllm \
#   --model google/gemma-2-9b-it \
#   --out vllm_results.csv

pixi run python corr.py \
  --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
  --perf-csv /nlpgpu/data/terry/ToolProj/src/exps_performance/out_hf_scale_deepseek_working/run_20250909_105603_results_seed_0.csv \
  --batch-size 1 \
  --vllm-tensor-parallel 8 \
  --vllm-gpu-mem-util 0.7 \
  --tb-dir  /nlpgpu/data/terry/ToolProj/src/exps_performance/out_hf_scale_deepseek_working/tb \
  --out softcorr_from_perf.csv

# correct_nl, correct_code, correct_code_exec
