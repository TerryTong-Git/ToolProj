pixi run  python lr_concept_features.py --tbdir /nlpgpu/data/terry/ToolProj/src/exps_performance/out_hf_scale_deepseek_working/tb --rep code \
    --label gamma --bits

pixi run  python lr_concept_features.py --tbdir /nlpgpu/data/terry/ToolProj/src/exps_performance/out_hf_scale_deepseek_working/tb --rep nl \
    --label gamma --bits

pixi run  python lr_concept_features.py --tbdir /nlpgpu/data/terry/ToolProj/src/exps_performance/out_hf_scale_deepseek_working/tb --rep code \
    --label gamma --bits --feats hf-cls --embed-model google-bert/bert-base-uncased

pixi run  python lr_concept_features.py --tbdir /nlpgpu/data/terry/ToolProj/src/exps_performance/out_hf_scale_deepseek_working/tb --rep nl \
    --label gamma --bits --feats hf-cls --embed-model google-bert/bert-base-uncased