MODELS=(meta-llama/Llama-3.1-8B-Instruct)
for MODEL in ${MODELS[@]}; do
    pixi run lm_eval --model vllm \
        --model_args pretrained=${MODEL},tensor_parallel_size=8,dtype=float16,gpu_memory_utilization=0.9,max_model_len=4096,download_dir=../models \
        --tasks ifeval,bbh,humaneval \
        --batch_size auto \
        --use_cache ./cache_dir \
        --output_path results/ \
        --confirm_run_unsafe_code True
done