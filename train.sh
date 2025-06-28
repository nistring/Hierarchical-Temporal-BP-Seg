#!/bin/bash

# Iterate over the config files by their numbers

for i in 1_GE 1_mind; do
    config_file="configs/config$i.yaml"
    echo "Running training with config: $config_file"
    python main.py --config_file "$config_file"
done
