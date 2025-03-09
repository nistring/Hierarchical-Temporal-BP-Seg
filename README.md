# Video object segmentation for Enhanced Brachial Plexus Visualization in Ultrasound Imaging

This project develops a video object segmentation system using a temporal segmentation model, specifically tailored for ultrasound images. The model integrates a segmentation architecture with temporal modeling capabilities to analyze sequences of images effectively. The goal is to apply this video object segmentation model to a **sequential ultrasound imaging technique (SUIT)** to recognize individual components of the **brachial plexus (BP)**, which is expected to aid in brachial plexus block procedures.

## Preview
![Preview](https://github.com/user-attachments/assets/863e9dac-49f6-47c3-8309-57d3a7c038df)

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd temporal-segmentation-project
pip install -r requirements.txt
```

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
