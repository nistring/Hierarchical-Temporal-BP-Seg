for i in sepGRU sepGRU2 sepGRU3 sepGRU4 sepGRU5 sepGRU6 sepGRU7; do
    config_file="configs/$i.yaml"
    echo "Running training with config: $config_file"
    python3 main.py --config_file "$config_file"
done
