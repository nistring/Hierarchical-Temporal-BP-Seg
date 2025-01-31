import cv2
from pathlib import Path

def get_max_frame_count(directory: str) -> int:
    """
    Get the maximum frame count from all video files in the specified directory.

    Args:
        directory (str): Path to the directory containing video files.

    Returns:
        int: Maximum frame count among all video files.
    """
    frame_counts = []
    for vid in Path(directory).rglob('*.mp4'):
        cap = cv2.VideoCapture(str(vid))
        frame_counts.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return max(frame_counts)

if __name__ == "__main__":
    print(get_max_frame_count('data/raw'))