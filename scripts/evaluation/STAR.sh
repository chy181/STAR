gpus=0
python -u ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" \
    --model-name "pre_train.DadaModel" \
    --data-name-list "Genesis.csv" \
    --model-hyper-params '{"K": 7, "alpha": 64, "batch_size": 128, "d_model": 32, "horizon": 1, "is_train": 0, "lr": 0.0005, "norm": true, "num_experts": 25, "num_shared_experts": 3, "plugin_lr": 0.003, "rank": 2, "sampling_rate": 0.1, "score_alpha": 1, "seq_len": 100, "backbone":"DADA"}' \
    --adapter "PreTrain_plugin_adapter" --gpus $gpus --num-workers 1 --timeout 60000 --save-path "adapter/Genesis/score/DadaModel"
python -u ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" \
    --model-name "pre_train.DadaModel" \
    --data-name-list "MSL.csv" \
    --model-hyper-params '{"K": 7, "alpha": 64, "batch_size": 128, "d_model": 32, "horizon": 1, "is_train": 0, "lr": 0.0005, "norm": true, "num_experts": 25, "num_shared_experts": 5, "plugin_lr": 0.003, "rank": 2, "sampling_rate": 0.1, "score_alpha": 1, "seq_len": 100, "backbone":"DADA"}' \
    --adapter "PreTrain_plugin_adapter" --gpus $gpus --num-workers 1 --timeout 60000 --save-path "adapter/MSL/score/DadaModel"
python -u ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" \
    --model-name "pre_train.UniTS" \
    --data-name-list "Genesis.csv" \
    --model-hyper-params '{"K": 7, "alpha": 64, "batch_size": 64, "d_model": 32, "horizon": 1, "is_train": 0, "lr": 0.0005, "norm": true, "num_experts": 25, "num_shared_experts": 7, "plugin_lr": 0.03, "rank": 2, "sampling_rate": 0.1, "score_alpha": 1, "seq_len": 96, "backbone":"UniTS"}' \
    --adapter "PreTrain_plugin_adapter" --gpus $gpus --num-workers 1 --timeout 60000 --save-path "adapter/Genesis/score/UniTS"
python -u ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" \
    --model-name "pre_train.UniTS" \
    --data-name-list "MSL.csv" \
    --model-hyper-params '{"K": 7, "alpha": 64, "batch_size": 64, "d_model": 32, "horizon": 1, "is_train": 0, "lr": 0.0005, "norm": true, "num_experts": 25, "num_shared_experts": 13, "plugin_lr": 0.003, "rank": 2, "sampling_rate": 0.1, "score_alpha": 1, "seq_len": 96, "backbone":"UniTS"}' \
    --adapter "PreTrain_plugin_adapter" --gpus $gpus --num-workers 1 --timeout 60000 --save-path "adapter/MSL/score/UniTS"
