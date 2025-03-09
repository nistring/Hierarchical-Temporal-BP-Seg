import cv2
import torch
from tqdm import tqdm
from src.model import TemporalSegmentationModel
from src.utils import process_video_stream, load_model
from torchvision.transforms import v2
import yaml

def main(
    video_source,
    checkpoint_path,
    output_path=None,
    num_classes=10,
    encoder_name="resnet34",
    segmentation_model_name="Unet",
    temporal_model="ConvGRU",
    image_size=(480, 640),
):
    """
    Main function to process video stream and perform segmentation.

    Args:
        video_source (str): Path to video file or camera index.
        checkpoint_path (str): Path to model checkpoint.
        output_path (str, optional): Path to save processed video. Default is None.
        num_classes (int): Number of classes for segmentation. Default is 10.
        encoder_name (str): Encoder model name. Default is "resnet34".
        segmentation_model_name (str): Segmentation model name. Default is "Unet".
        temporal_model (str): Temporal model type. Default is "ConvGRU".
        image_size (tuple): Image size for the model. Default is (480, 640).
    """
    # Load the model with the specified parameters and checkpoint
    model = load_model(
        TemporalSegmentationModel(encoder_name, segmentation_model_name, num_classes, image_size, temporal_model), checkpoint_path
    )
    model = model.cuda()  # Move model to GPU

    # Check if video_source is a digit (camera index)
    if video_source.isdigit():
        video_source = int(video_source)

    # Open video capture
    cap = cv2.VideoCapture(video_source)

    # Initialize video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing Video")  # Progress bar

    hidden_state = None  # Initialize hidden state for temporal model

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale and then to tensor
        frame = v2.ToImage()(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).cuda()
        tensor_input = v2.ToDtype(torch.float32, scale=True)(v2.Resize(image_size)(frame))[
            None, None
        ]  # Add batch and sequence dimensions and move to GPU

        # Perform inference without gradient calculation
        with torch.no_grad():
            output, hidden_state = model(tensor_input, hidden_state)

        # Apply softmax to get segmentation masks
        masks = torch.nn.Softmax(dim=2)(output)[0, 0]

        # Process the frame with the segmentation masks
        processed_frame = process_video_stream(frame, masks)

        # Write the processed frame to output video or display it
        if output_path:
            out.write(processed_frame)
        else:
            cv2.imshow("Segmentation", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        pbar.update(1)  # Update progress bar

    pbar.close()
    cap.release()  # Release video capture
    if output_path:
        out.release()  # Release video writer
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segmentation Demo")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--video_source", type=str, required=True, help="Path to video file or camera index")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, help="Path to save processed video")

    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Call main function with parameters from config and command line arguments
    main(
        args.video_source,
        args.checkpoint,
        args.output,
        config['model'].get('num_classes', 10),
        config['model'].get('encoder_name', "resnet34"),
        config['model'].get('segmentation_model_name', "Unet"),
        config['model'].get('temporal_model', "ConvGRU"),
        tuple(config['model'].get('image_size', (480, 640))),
    )
