pixi run vllm serve "mistralai/Mistral-Small-24B-Instruct-2501" \
                    --dtype "float16" \
                    -tp 8 \
                    --gpu_memory_utilization 0.95 \
                    --max-model-len 8192 \
                    --download_dir "/nlpgpu/data/terry/ToolProj/src/models" \