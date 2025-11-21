pixi run vllm serve google/gemma-2-9b-it \
                    --dtype "float16" \
                    -tp 8 \
                    --gpu_memory_utilization 0.95 \
                    --max-model-len 8192 \
                    --download_dir "/nlpgpu/data/terry/ToolProj/src/models" \