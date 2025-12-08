CSV="/nlpgpu/data/terry/ToolProj/src/exps_performance/results/ministral-14b-2512_seed0/tb/run_20251205_194501/res.csv"
uv run --no-sync python -m src.exps_logistic.main --csv "$CSV" --rep code --device cuda --hf-batch 8 --logreg-c-grid 0.25 0.5 1 2 4 --logreg-max-iter-grid 100 200 400 --logreg-cv-folds 5 \
    --label gamma --bits --feats hf-cls --embed-model microsoft/codebert-base --max_iter 20 --C 0.5 

uv run --no-sync python -m src.exps_logistic.main --csv "$CSV" --rep nl --device cuda --hf-batch 1 --logreg-c-grid 0.25 0.5 1 2 4 --logreg-max-iter-grid 100 200 400 --logreg-cv-folds 5 \
    --label gamma --bits --feats hf-cls --embed-model microsoft/codebert-base --max_iter 20 --C 0.5 

uv run --no-sync python -m src.exps_logistic.main --csv "$CSV" --rep code --device cuda  \
    --label gamma --bits --feats tfidf 

uv run --no-sync python -m src.exps_logistic.main --csv "$CSV" --rep nl --device cuda  \
    --label gamma --bits --feats tfidf 

# neulab/codebert-python