#!/bin/bash

# Array of config files
configs=("sepGRU384_2")

# Run tests in parallel on different GPUs (max 4 GPUs at a time)
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    gpu=$((i % 4))  # Cycle through GPUs 0-3
    
    # Wait if we've reached the GPU limit (every 4 jobs)
    if (( i > 0 && i % 4 == 0 )); then
        wait
    fi
    
    # Set CUDA_VISIBLE_DEVICES to isolate GPU usage
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