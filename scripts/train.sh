for i in convGRU sepGRU sepGRU_batch1 sepGRU_batch2 sepGRU_batch3 sepGRU_batch4; do
    config_file="configs/$i.yaml"
    echo "Running training with config: $config_file"
    python3 main.py --config_file "$config_file"
done
