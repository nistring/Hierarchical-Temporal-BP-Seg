for dataset in "GE" "mindray" "val"; do
    for i in 20; do
        python main.py --config_file lightning_logs/config$i/config.yaml \
            --mode "test" \
            --best_model_path lightning_logs/config$i/checkpoints/last.ckpt \
            --test_data_path ./data/SUIT/images/$dataset \
            --test_annotations_path ./data/SUIT/coco_annotations/${dataset}_updated.json
    done
done