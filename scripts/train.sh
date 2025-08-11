for i in "depth384" "depth416" "depth448" "point384" "point416" "point448" "reduced416"; do
    config_file="configs/$i.yaml"
    echo "Running training with config: $config_file"
    python3 main.py --config_file "$config_file"
done