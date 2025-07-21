#!/bin/bash

# Array of config files
configs=("no_temp" "sepGRU")

# Run tests in parallel on different GPUs
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    gpu=$i
    
    echo "Starting test for $config on GPU $gpu"
    python3 main.py --config_file lightning_logs/$config/config.yaml \
        --mode "test" \
        --best_model_path lightning_logs/$config/checkpoints/last.ckpt \
        --test_data_path ./data/SUIT/images/val \
        --test_annotations_path ./data/SUIT/coco_annotations/val_updated.json \
        --gpu $gpu &
done

# Wait for all background processes to complete
wait
echo "All tests completed"