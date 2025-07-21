for i in attnGRU sepGRU convGRU no_temp; do
    config_file="configs/$i.yaml"
    echo "Running training with config: $config_file"
    python3 main.py --config_file "$config_file"
done
