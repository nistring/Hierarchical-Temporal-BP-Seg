# Real-Time Brachial Plexus Ultrasound Segmentation Using Lightweight Hierarchical Temporal Fusion

This project develops a video object segmentation system using a temporal segmentation model, specifically tailored for ultrasound images. The model integrates a segmentation architecture with temporal modeling capabilities to analyze sequences of images effectively. The goal is to apply this video object segmentation model to a **sequential ultrasound imaging technique (SUIT)** to recognize individual components of the **brachial plexus (BP)**, which is expected to aid in brachial plexus block procedures.

<img width="3033" height="1485" alt="image" src="https://github.com/user-attachments/assets/3264860a-eb0e-415a-b386-291501a08041" />

[Preprint](https://www.techrxiv.org/doi/full/10.36227/techrxiv.176117951.13856310/v1)

## Note
This is an advanced study building on our [previous work](https://github.com/nistring/Ultrasound-Optimal-View-Detection) in ultrasound image classification and segmentation. Data are currently unavailable, but we plan to upload them soon. This model is most suitable for use with the _Sonosite_ ultrasound machine. We have newly opened a real-time NPU-accelerated Android app! (https://github.com/nistring/SUIT-app). I apologize for not providing detailed documentation on this. Feel free to contact us should you have any questions. Thanks!

## Preview
[Youtube link](https://www.youtube.com/watch?v=nb6DnPcaAVo)

## Architecture

The model (`src/model.py`) combines three components:

1. **Encoder** — a segmentation backbone from [`segmentation_models_pytorch`](https://github.com/qubvel-org/segmentation_models.pytorch) (e.g. Segformer `mit_b0`, `efficientnet-b0`, `resnet18`).
2. **Temporal modules** — convolutional RNNs (`ConvGRU`, `ConvLSTM`, `ConvRNN`, and lightweight variants in `src/temp_module/`) inserted at the deepest encoder feature levels. With hierarchical fusion enabled, deeper temporal features are upsampled and concatenated into shallower levels.
3. **Decoder + head** — Segformer / UNet / DeepLabV3+ from `segmentation_models_pytorch`.

Inputs are single-channel (grayscale) ultrasound frames (default 416×416); outputs are `num_classes + 1` channels (background + 8 brachial-plexus structures). Training uses truncated backpropagation-through-time (TBPTT) over image sequences.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd Hierarchical-Temporal-BP-Seg
pip install -r requirements.txt
```

A pretrained weight is provided in the `lightning_logs` directory. Use git lfs to download it.
Compiled weights can be downloaded from a [Google Drive](https://drive.google.com/drive/folders/1o2LHaAs774_LCP0G5KFya3torGOPq4aK?usp=sharing) folder that supports Galaxy Tab S8, S9, S10, and S11.

## Data

- Grayscale ultrasound images go in `data/SUIT/images/`.
- COCO-style annotations go in `data/SUIT/coco_annotations/`, grouped by `video_id`.

Segmentation masks are produced with a two-stage auto-labeling pipeline: human keyframe bounding boxes are propagated to per-frame bbox seeds, which are then converted to per-frame polygon masks by a promptable segmenter — either **SAM2** (`data/SUIT/autolabel.py`) or **UltraSam** (`data/SUIT/autolabel_ultrasam.py`), the latter being more robust to empty masks on ultrasound.

## Usage

### Training

```bash
python main.py --config_file configs/<config>.yaml
```

Experiment configurations live in `configs/`. Multi-GPU DDP is set in the YAML (`trainer.gpus`); add `--gpu 0` to force a single GPU. The `*_ultrasam.yaml` variants train on UltraSam-generated labels.

### Testing

```bash
python main.py --config_file lightning_logs/<version>/config.yaml \
    --mode test \
    --best_model_path lightning_logs/<version>/checkpoints/last.ckpt \
    --test_data_path ./data/SUIT/images/sonosite_val \
    --test_annotations_path ./data/SUIT/coco_annotations/sonosite_val_sam2.json \
    --gpu 0
```

### Demo (video inference)

```bash
python demo.py --config <config.yaml> --input_folder <video_dir_or_camera_idx> \
    --output_folder <output_dir> --checkpoint <ckpt_path> --gpu 0
```

### On-device export

Export a single-frame, stateful inference model for mobile NPUs:

```bash
python deploy/export.py --config configs/seq50_relu.yaml --device-name "Galaxy Tab S10"
```

Supported devices: Galaxy Tab S8 / S9 (Qualcomm, via AI Hub) and S10 / S11 (MediaTek, via LiteRT AOT compilation). The `seq50_relu` variant uses ReLU activations, which are better supported than GeLU on mobile NPUs.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

This project is inspired by and builds upon the work done in the following repository:

- [Convolutional LSTM and GRU](https://github.com/aserdega/convlstmgru)
</content>
</invoke>
