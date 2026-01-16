RESULTS_DIR="/Users/terrytong/Documents/Projects/CCG/ToolProj/src/exps_performance/results"


MODELS=(
    # anthropic/claude-haiku-4.5
    # qwen/qwen-2.5-coder-32b-instruct
    # mistralai/ministral-14b-2512
    # meta-llama/llama-3.1-405b-instruct
    # openai/gpt-4o-mini
    # google/gemini-2.5-flash
    mistralai/codestral-2508
    mistralai/mixtral-8x22b-instruct
    google/gemini-2.5-flash
)
for SEED in {0..2}; do
    for MODEL in "${MODELS[@]}"; do
        MODELS_ARG=""
        if [ -n "$MODEL" ]; then
            MODELS_ARG="--models $MODEL"
        fi
    # uv run python -m src.exps_logistic.main \
    #     --rep code --label theta_new \
    #     --results-dir src/exps_performance/results \
    #     --device mps --hf-batch 8 \
    #     --feats hf-cls --embed-model google-bert/bert-base-uncased \
    #     --no-cv --max_iter 20 --C 0.5 \
    #     $MODELS_ARG \
    #     --seed $SEED

    # uv run python -m src.exps_logistic.main \
    #     --rep nl --label theta_new \
    #     --results-dir src/exps_performance/results \
    #     --device mps --hf-batch 8 \
    #     --feats hf-cls --embed-model google-bert/bert-base-uncased \
    #     --no-cv --max_iter 20 --C 0.5 \
    #     $MODELS_ARG \
    #     --seed $SEED

    uv run python -m src.exps_logistic.main \
        --rep code --label theta_new \
        --results-dir src/exps_performance/results \
        --device mps --hf-batch 8 \
        --feats openrouter --embed-model openai/text-embedding-3-large \
        --no-cv --max_iter 20 --C 0.5 \
        $MODELS_ARG \
        --seed $SEED

    uv run python -m src.exps_logistic.main \
        --rep nl --label theta_new \
        --results-dir src/exps_performance/results \
        --device mps --hf-batch 8 \
        --feats openrouter --embed-model openai/text-embedding-3-large \
        --no-cv --max_iter 20 --C 0.5 \
        $MODELS_ARG \
        --seed $SEED

        # uv run --no-sync python -m src.exps_logistic.main --results-dir "$RESULTS_DIR" $MODELS_ARG --rep code --device mps --hf-batch 8  \
        #     --label gamma --no-cv  --bits --feats hf-cls --embed-model google-bert/bert-base-uncased --max_iter 20 --C 0.5 --seed $SEED --kinds-preset extended

        # uv run --no-sync python -m src.exps_logistic.main --results-dir "$RESULTS_DIR" $MODELS_ARG --rep nl --device mps --hf-batch 1  \
        #     --label gamma --no-cv --bits --feats hf-cls --embed-model google-bert/bert-base-uncased --max_iter 20 --C 0.5 --seed $SEED --kinds-preset extended

        # uv run --no-sync python -m src.exps_logistic.main --results-dir "$RESULTS_DIR" $MODELS_ARG --rep code --device mps  \
        #     --label gamma --no-cv --bits --feats tfidf --seed $SEED

        # uv run --no-sync python -m src.exps_logistic.main --results-dir "$RESULTS_DIR" $MODELS_ARG --rep nl --device mps  \
        #     --label gamma --no-cv --bits --feats tfidf --seed $SEED
    done
done