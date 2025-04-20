import cv2
import torch
from tqdm import tqdm
from src.model import TemporalSegmentationModel
from src.utils import process_video_stream, load_model, post_processing
from torchvision.transforms import v2
import yaml


def process_video(video_source, output_path, model, image_size):
    """
    Process a single video file or stream and perform segmentation.

    Args:
        video_source (str or int): Path to video file or camera index.
        output_path (str): Path to save processed video.
        model (torch.nn.Module): Loaded segmentation model.
        image_size (tuple): Image size for the model.
    """
    cap = cv2.VideoCapture(video_source)

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing {video_source}")  # Progress bar

    hidden_state = None  # Initialize hidden state for temporal model

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = v2.ToImage()(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).cuda()
        tensor_input = v2.ToDtype(torch.float32, scale=True)(v2.Resize(image_size)(frame))[
            None, None
        ]  # Add batch and sequence dimensions and move to GPU

        with torch.no_grad():
            output, hidden_state = model(tensor_input, hidden_state)

        output = post_processing(output)[0]
        frame = process_video_stream(frame, output)

        if output_path:
            out.write(frame)
        else:
            cv2.imshow("Segmentation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        pbar.update(1)  # Update progress bar

    pbar.close()
    cap.release()  # Release video capture
    if output_path:
        out.release()  # Release video writer
    cv2.destroyAllWindows()  # Close all OpenCV windows


def main(
    input_folder,
    output_folder,
    checkpoint_path,
    num_classes=10,
    encoder_name="resnet34",
    segmentation_model_name="Unet",
    temporal_model="ConvGRU",
    image_size=(480, 640),
    temporal_depth=1,  # Add temporal_depth parameter
    attention_module=None,
):
    """
    Main function to process video stream and perform segmentation.

    Args:
        input_folder (str): Path to input folder containing video files.
        output_folder (str): Path to output folder to save processed videos.
        checkpoint_path (str): Path to model checkpoint.
        num_classes (int): Number of classes for segmentation. Default is 10.
        encoder_name (str): Encoder model name. Default is "resnet34".
        segmentation_model_name (str): Segmentation model name. Default is "Unet".
        temporal_model (str): Temporal model type. Default is "ConvGRU".
        image_size (tuple): Image size for the model. Default is (480, 640).
        temporal_depth (int): Depth of the temporal model. Default is 1.
    """
    # Load the model with the specified parameters and checkpoint
    model = load_model(
        TemporalSegmentationModel(
            encoder_name, segmentation_model_name, num_classes, image_size, temporal_model, temporal_depth=temporal_depth,
        ),
        checkpoint_path,
    )

    # Check if input_folder is a digit (camera index)
    if input_folder.isdigit():
        input_folder = int(input_folder)
        process_video(input_folder, None, model, image_size)

    # Process each video file in the input folder
    for video_file in os.listdir(input_folder):
        video_source = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)
        process_video(video_source, output_path, model, image_size)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Segmentation Demo")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input folder containing video files")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder to save processed videos")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Call main function with parameters from config and command line arguments
    main(
        args.input_folder,
        args.output_folder,
        args.checkpoint,
        config["model"].get("num_classes", 10),
        config["model"].get("encoder_name", "resnet34"),
        config["model"].get("segmentation_model_name", "Unet"),
        config["model"].get("temporal_model", "ConvGRU"),
        tuple(config["model"].get("image_size", (480, 640))),
        config["model"].get("temporal_depth", 1),  # Add temporal_depth parameter
        config["model"].get("attention_module", None),
    )
