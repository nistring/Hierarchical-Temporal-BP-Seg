for run in {1..4}; do
    echo "Run $run"
    for i in "seplstm" "sepgru" "reduced" "minlstm" "mingru" "convrnn" "convlstm" "convgru"; do
        config_file="configs/$i.yaml"
        echo "Running training with config: $config_file"
        python3 main.py --config_file "$config_file"
    done
done