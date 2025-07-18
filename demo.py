import cv2
import torch
from tqdm import tqdm
from src.model import TemporalSegmentationModel
from src.utils import process_video_stream, load_model, post_processing
from torchvision.transforms import v2
import yaml
import os


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


def main(config, input_folder, output_folder, checkpoint_path):
    """
    Main function to process video stream and perform segmentation.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        input_folder (str): Path to input folder containing video files.
        output_folder (str): Path to output folder to save processed videos.
        checkpoint_path (str): Path to model checkpoint.
    """
    # Initialize the model with config parameters
    model_config = {
        "encoder_name": config["model"]["encoder_name"],
        "segmentation_model_name": config["model"]["segmentation_model_name"],
        "num_classes": config["model"]["num_classes"],
        "temporal_model": config["model"]["temporal_model"],
        "image_size": tuple(config["model"]["image_size"]),
        "encoder_depth": config["model"]["encoder_depth"],
        "temporal_depth": config["model"]["temporal_depth"],
        "freeze_encoder": config["model"].get("freeze_encoder", False),
        "num_layers": config["model"].get("num_layers", 1),
    }
    
    # Add any additional model arguments from config
    if "model_kwargs" in config["model"]:
        model_config.update(config["model"]["model_kwargs"])
    
    # Load the model with the specified parameters and checkpoint
    model = load_model(
        TemporalSegmentationModel(**model_config),
        checkpoint_path,
    ).cuda()

    # Check if input_folder is a digit (camera index)
    if str(input_folder).isdigit():
        input_folder = int(input_folder)
        process_video(input_folder, None, model, tuple(config["model"]["image_size"]))
        return

    # Process each video file in the input folder
    for video_file in os.listdir(input_folder):
        video_source = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)
        process_video(video_source, output_path, model, tuple(config["model"]["image_size"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segmentation Demo")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input folder containing video files or camera index")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder to save processed videos")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Call main function with parameters from config and command line arguments
    main(config, args.input_folder, args.output_folder, args.checkpoint)