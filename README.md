# Real-Time Brachial Plexus Ultrasound Segmentation Using Lightweight Hierarchical Temporal Fusion

This project develops a video object segmentation system using a temporal segmentation model, specifically tailored for ultrasound images. The model integrates a segmentation architecture with temporal modeling capabilities to analyze sequences of images effectively. The goal is to apply this video object segmentation model to a **sequential ultrasound imaging technique (SUIT)** to recognize individual components of the **brachial plexus (BP)**, which is expected to aid in brachial plexus block procedures.

<img width="3033" height="1485" alt="image" src="https://github.com/user-attachments/assets/3264860a-eb0e-415a-b386-291501a08041" />

[Preprint](https://www.techrxiv.org/doi/full/10.36227/techrxiv.176117951.13856310/v1)

## Note
This is an advanced study building on our [previous work](https://github.com/nistring/Ultrasound-Optimal-View-Detection) in ultrasound image classification and segmentation. Data are currently unavailable, but we plan to upload them soon. This model is most suitable for use with the _Sonosite_ ultrasound machine. We aim to develop a real-time application in our upcoming research. Feel free to contact us should you have any questions. Thanks!

## Preview
[Youtube link](https://www.youtube.com/watch?v=nb6DnPcaAVo)

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd temporal-segmentation-project
pip install -r requirements.txt
```

A pretrained weight is provided in `lightning_logs` directory. Use git lfs to download it.

## Usage

To train the model, use the following command:

```bash
python src/train.py --config_file config.yaml
```

- Place your ultrasound images in the `data/SUIT/images` directory.
- Place your coco style annotations in the `data/SUIT/coco_annotations` directory.

To run a demo on a video stream, use:

```bash
python demo.py --video_source <video_path_or_camera_index> --checkpoint <checkpoint_path>
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

This project is inspired by and builds upon the work done in the following repository:

- [Convolutional LSTM and GRU](https://github.com/aserdega/convlstmgru)
