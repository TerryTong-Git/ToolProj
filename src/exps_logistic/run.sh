# pixi run  python lr_concept_features.py --tbdir /nlpgpu/data/terry/ToolProj/src/exps_performance/out_hf_scale_deepseek_working/tb --rep code \
#     --label gamma --bits --max_iter 20 --C 0.5

pixi run  python lr_concept_features.py --tbdir /nlpgpu/data/terry/ToolProj/src/exps_performance/out_hf_scale_deepseek_working/tb --rep nl \
    --label gamma --bits --max_iter 20 --C 0.5

# pixi run  python lr_concept_features.py --tbdir /nlpgpu/data/terry/ToolProj/src/exps_performance/out_hf_scale_deepseek_working/tb --rep code \
#     --label gamma --bits --feats hf-cls --embed-model google-bert/bert-base-uncased --max_iter 1000 --C 0.5

# pixi run  python lr_concept_features.py --tbdir /nlpgpu/data/terry/ToolProj/src/exps_performance/out_hf_scale_deepseek_working/tb --rep nl \
#     --label gamma --bits --feats hf-cls --embed-model google-bert/bert-base-uncased --max_iter 1000 --C 0.5