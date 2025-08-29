for run in {1..5}; do
    echo "Run $run"
    for i in "unettbptt" "deeplabtbptt"; do
        config_file="configs/$i.yaml"
        echo "Running training with config: $config_file"
        python3 main.py --config_file "$config_file"
    done
done