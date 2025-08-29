#!/bin/bash

CONFIG_DIR="configs"
INPUT_FOLDER="data/SUIT/demo/input/"
OUTPUT_BASE="data/SUIT/demo"
CHECKPOINT_BASE="lightning_logs"
CONFIG_FILE_NAME="config.yaml"
MAX_PARALLEL=40

i=0
for config_yaml in "$CONFIG_DIR"/*.yaml; do
    config_name=$(basename "$config_yaml" .yaml)
    gpu=$((i % 4))  # Cycle through GPUs 0-3

    # Wait if we've reached the parallel limit
    if (( i > 0 && i % MAX_PARALLEL == 0 )); then
        wait
    fi

    python demo.py --input_folder "$INPUT_FOLDER" \
                   --output_folder "$OUTPUT_BASE/$config_name/" \
                   --checkpoint "$CHECKPOINT_BASE/$config_name/checkpoints/last.ckpt" \
                   --config "$CHECKPOINT_BASE/$config_name/$CONFIG_FILE_NAME" \
                   --gpu $gpu &
    ((i++))
done

wait
echo "All demos completed"