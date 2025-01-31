#!/bin/bash

# Iterate over the config files by their numbers
for i in {5..11}; do
    config_file="configs/config$i.yaml"
    echo "Running training with config: $config_file"
    python _train.py --config_file "$config_file"
done
