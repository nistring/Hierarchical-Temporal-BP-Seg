for i in 1_46; do
    # Run GE_val on GPU 0
    python main.py --config_file lightning_logs/config$i/config.yaml \
        --mode "test" \
        --best_model_path lightning_logs/config$i/checkpoints/last.ckpt \
        --test_data_path ./data/SUIT/images/GE_val \
        --test_annotations_path ./data/SUIT/coco_annotations/GE_val_updated.json \
        --gpu 1 &
    
    # Run mindray_val on GPU 1
    python main.py --config_file lightning_logs/config$i/config.yaml \
        --mode "test" \
        --best_model_path lightning_logs/config$i/checkpoints/last.ckpt \
        --test_data_path ./data/SUIT/images/mindray_val \
        --test_annotations_path ./data/SUIT/coco_annotations/mindray_val_updated.json \
        --gpu 2 &
    
    # Run val on GPU 2
    python main.py --config_file lightning_logs/config$i/config.yaml \
        --mode "test" \
        --best_model_path lightning_logs/config$i/checkpoints/last.ckpt \
        --test_data_path ./data/SUIT/images/val \
        --test_annotations_path ./data/SUIT/coco_annotations/val_updated.json \
        --gpu 3
    
    # Wait for all background processes to complete
    wait
done