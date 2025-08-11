config="sepGRU416_2"
python demo.py --input_folder data/SUIT/demo/input/ \
                --output_folder data/SUIT/demo/$config/ \
                --checkpoint lightning_logs/$config/checkpoints/last.ckpt \
                --config lightning_logs/$config/config.yaml