RESULTS_DIR="/nlpgpu/data/terry/ToolProj/src/exps_performance/results"


MODELS=(
    anthropic/claude-haiku-4.5
    qwen/qwen-2.5-coder-32b-instruct
    mistralai/ministral-14b-2512
    meta-llama/llama-3.1-405b-instruct
    openai/gpt-4o-mini
    google/gemini-2.5-flash
)
for SEED in {0..2}; do
    for MODEL in "${MODELS[@]}"; do
        MODELS_ARG=""
        if [ -n "$MODEL" ]; then
            MODELS_ARG="--models $MODEL"
        fi

        uv run --no-sync python -m src.exps_logistic.main --results-dir "$RESULTS_DIR" $MODELS_ARG --rep code --device cuda --hf-batch 8  \
            --label gamma --no-cv  --bits --feats hf-cls --embed-model google-bert/bert-base-uncased --max_iter 20 --C 0.5 --seed $SEED --kinds-preset extended

        uv run --no-sync python -m src.exps_logistic.main --results-dir "$RESULTS_DIR" $MODELS_ARG --rep nl --device cuda --hf-batch 1  \
            --label gamma --no-cv --bits --feats hf-cls --embed-model google-bert/bert-base-uncased --max_iter 20 --C 0.5 --seed $SEED --kinds-preset extended

        # uv run --no-sync python -m src.exps_logistic.main --results-dir "$RESULTS_DIR" $MODELS_ARG --rep code --device cuda  \
        #     --label gamma --no-cv --bits --feats tfidf --seed $SEED

        # uv run --no-sync python -m src.exps_logistic.main --results-dir "$RESULTS_DIR" $MODELS_ARG --rep nl --device cuda  \
        #     --label gamma --no-cv --bits --feats tfidf --seed $SEED
    done
done
# neulab/codebert-python