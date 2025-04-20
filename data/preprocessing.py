import cv2
import os
import xml.etree.ElementTree as ET
import json

thresh_binary = 20
kernel_size = 8
vid_root = "raw/videos"


def crop(frame, roi=None):
    """
    Crop the frame to the bounding box of the largest contour.

    Args:
        frame (numpy.ndarray): The input video frame.
        roi (tuple): The region of interest (x, y, w, h) to crop the frame.

    Returns:
        tuple: The bounding box (x, y, w, h) of the largest contour.
    """
    if roi is not None:
        frame = frame[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
    original_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    img = cv2.threshold(original_img, thresh_binary, 255, cv2.THRESH_TOZERO)[1]
    ret = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if roi is not None:
        x += roi[0]
        y += roi[1]

    return (x, y, w, h)


def crop_vid(name, roi=None):
    """
    Crop the video to the bounding box of the largest contour in the first frame.

    Args:
        name (str): The name of the video file.
    """
    bbox = None
    cap = cv2.VideoCapture(os.path.join(vid_root, name))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if bbox is None:
            bbox = crop(frame, roi)
            if name == "00610090.mp4":
                bbox = (639, 130, 819, 747)
            out = cv2.VideoWriter(f"SUIT/demo/input/{name}", fourcc, cap.get(cv2.CAP_PROP_FPS), (bbox[2], bbox[3]))
        out.write(frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]])
    cap.release()
    out.release()
    print(name, "done")


def preprocess(anno_root, roi=None):
    """
    Preprocess the video frames and annotations.

    Args:
        is_vid_train_frame (bool): Flag to indicate if the frames are for training or validation.
    """
    anno_dir = anno_root.split("/")[-1]
    is_vid_train_frame = anno_dir == "train"
    ann_file = f"SUIT/coco_annotations/{anno_dir}.json"
    save_dir = f"SUIT/images/{anno_dir}"
    os.makedirs(save_dir, exist_ok=True)

    categories = [
        {"id": 1, "name": "C5"},
        {"id": 2, "name": "C6"},
        {"id": 3, "name": "C7"},
        {"id": 4, "name": "C8"},
        {"id": 5, "name": "UT"},
        {"id": 6, "name": "MT"},
        {"id": 7, "name": "LT"},
        {"id": 8, "name": "SSN"},
        {"id": 9, "name": "AD"},
        {"id": 10, "name": "PD"},
    ]
    cat_list = [cat["name"] for cat in categories]
    videos, images, annotations = [], [], []
    cat_id, file_id, img_id, anno_id = 1, 1, 1, 1

    for vid_id, anno_dir in enumerate(os.listdir(anno_root), start=1):
        tree = ET.parse(os.path.join(anno_root, anno_dir, "annotations.xml"))
        root = tree.getroot()
        name = anno_dir + ".mp4"
        first_frame_img_id = img_id
        print(name)

        videos.append(dict(id=vid_id, name=name))

        bbox = None
        cap = cv2.VideoCapture(os.path.join(vid_root, name))
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if bbox is None:
                bbox = crop(frame, roi)
                if anno_dir == "00610090":
                    bbox = (639, 130, 819, 747)
            file_name = f"image_{img_id}.png"
            cv2.imwrite(os.path.join(save_dir, file_name), frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]])
            images.append(dict(file_name=file_name, height=bbox[3], width=bbox[2], id=img_id, video_id=vid_id, frame_id=frame_id))
            img_id += 1
            frame_id += 1
        cap.release()

        for track in root.findall("./track"):
            cat_name = track.attrib["label"].upper()
            for box in track:
                attr = box.attrib
                if not bool(int(attr["outside"])):
                    width = float(attr["xbr"]) - float(attr["xtl"])
                    height = float(attr["ybr"]) - float(attr["ytl"])
                    annotations.append(
                        dict(
                            id=anno_id,
                            image_id=first_frame_img_id + int(attr["frame"]) - int(attr["keyframe"]),
                            video_id=vid_id,
                            category_id=cat_list.index(cat_name) + 1,
                            instance_id=1,
                            bbox=[float(attr["xtl"]) - bbox[0], float(attr["ytl"]) - bbox[1], width, height],
                            area=width * height,
                            occluded=bool(int(attr["occluded"])),
                            truncated=False,
                            iscrowd=False,
                            is_vid_train_frame=is_vid_train_frame,
                            visibility=1.0,
                        )
                    )
                    anno_id += 1

    with open(ann_file, "w") as f:
        json.dump(dict(categories=categories, videos=videos, images=images, annotations=annotations), f)


if __name__ == "__main__":
    # preprocess("raw/anno/GE", roi=(500, 100, 1100, 800))
    # preprocess("raw/anno/mindray")

    # for i in range(51, 76):
    #     crop_vid(f"CNUH_DC04_BPB1_00{str(i)}.mp4")
    for vid in os.listdir("raw/anno/mindray"):
        crop_vid(vid + ".mp4")
    for vid in os.listdir("raw/anno/GE"):
        crop_vid(vid + ".mp4", roi=(500, 100, 1100, 800))