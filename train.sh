for i in 1_31; do
    config_file="configs/config$i.yaml"
    echo "Running training with config: $config_file"
    python main.py --config_file "$config_file"
done
