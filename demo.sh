python demo.py --video_source data/SUIT/demo/input/CNUH_DC04_BPB1_0041.mp4 \
                   --output data/SUIT/demo/output/CNUH_DC04_BPB1_0041.mp4 \
                   --checkpoint lightning_logs/version_0/checkpoints/epoch=56-step=570.ckpt\
                   --num_classes 10 \
                   --encoder_name resnet34 \
                   --segmentation_model_name Unet \
                   --temporal_model ConvGRU