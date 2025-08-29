import cv2
import torch
from tqdm import tqdm
from src.model import TemporalSegmentationModel
from src.utils import process_video_stream, load_model, post_processing
from torchvision.transforms import v2
import yaml
import os


def process_video(video_source, output_path, model, image_size, device):
    cap = cv2.VideoCapture(video_source)
    
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing {video_source}")
    hidden_state = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = v2.ToImage()(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).to(device)
        tensor_input = v2.ToDtype(torch.float32, scale=True)(v2.Resize(image_size)(frame))[None, None]

        with torch.no_grad():
            output, hidden_state = model(tensor_input, hidden_state)

        output = post_processing(output)[0]
        frame = process_video_stream(frame, output)

        if out:
            out.write(frame)
        else:
            cv2.imshow("Segmentation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        pbar.update(1)

    pbar.close()
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


def main(config, input_folder, output_folder, checkpoint_path, gpu):
    os.makedirs(output_folder, exist_ok=True)
    model_config = config["model"].copy()
    model_config["image_size"] = tuple(model_config["image_size"])
    if "model_kwargs" in model_config:
        model_config.update(model_config.pop("model_kwargs"))
    if "kernel_size" in model_config:
        model_config["kernel_size"] = tuple(model_config["kernel_size"])
    
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = load_model(TemporalSegmentationModel(**model_config), checkpoint_path).to(device)

    if str(input_folder).isdigit():
        process_video(int(input_folder), None, model, model_config["image_size"], device)
        return

    for video_file in os.listdir(input_folder):
        video_source = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)
        process_video(video_source, output_path, model, model_config["image_size"], device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segmentation Demo")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input folder containing video files or camera index")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder to save processed videos")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id to use")

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    main(config, args.input_folder, args.output_folder, args.checkpoint, args.gpu)