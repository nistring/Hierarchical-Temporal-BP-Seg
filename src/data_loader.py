import random
from typing import Optional, List, Iterator

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision import tv_tensors
from pycocotools.coco import COCO

# Mapping of category IDs to new category IDs
category_match = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 3, # MT -> C7
    7: 4, # LT -> C8
    8: 6,
    9: 7,
    10: 8,
}

# category_match = {
#     1: 1,
#     2: 2,
#     3: 3,
#     4: 4,
#     5: 5,
#     6: 3, # MT -> C7
#     7: 4, # LT -> C8
#     8: 5,
#     9: 5,
#     10: 5,
# }

# category_match = {
#     1: 1,
#     2: 1,
#     3: 2,
#     4: 3,
#     5: 1,
#     6: 2, # MT -> C7
#     7: 3, # LT -> C8
#     8: 1,
#     9: 1,
#     10: 1,
# }

class BaseUltrasoundDataset(Dataset):
    def __init__(self, data_dir: str, annotations_file: str, image_size: tuple, sequence_length: int, batch_size: int):
        """
        Initialize the BaseUltrasoundDataset.

        Args:
            data_dir (str): Directory containing the data.
            annotations_file (str): Path to the annotations file.
            image_size (tuple): Size of the input image.
            sequence_length (int): Sequence length.
            batch_size (int): Batch size.
        """
        self.data_dir = data_dir
        self.coco = COCO(annotations_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.num_classes = 10
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.video_id_to_img_ids = self._group_images_by_video()
        self.preprocess = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(image_size),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def _group_images_by_video(self) -> dict:
        """
        Group images by video.

        Returns:
            dict: Dictionary mapping video IDs to image IDs.
        """
        video_id_to_img_ids = {}
        for img_id in self.ids:
            video_id = self.coco.loadImgs(img_id)[0]["video_id"]
            if video_id not in video_id_to_img_ids:
                video_id_to_img_ids[video_id] = []
            video_id_to_img_ids[video_id].append(img_id)
        return video_id_to_img_ids

    def _load_image_and_mask(self, img_id: int) -> tuple:
        """
        Load an image and its corresponding mask.

        Args:
            img_id (int): Image ID.

        Returns:
            tuple: Loaded image and mask.
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]["file_name"]

        image = decode_image(str(self.data_dir / path), mode="GRAY")
        mask = torch.zeros(image.shape[1], image.shape[2], dtype=torch.uint8)

        for ann in coco_annotation:
            ann_mask = torch.tensor(self.coco.annToMask(ann), dtype=torch.bool)
            mask[ann_mask] = category_match[ann["category_id"]]
            # mask[ann_mask] = ann["category_id"]

        return image, mask

    def _load_images_and_masks(self, img_ids: list) -> tuple:
        """
        Load images and their corresponding masks.

        Args:
            img_ids (list): List of image IDs.

        Returns:
            tuple: Loaded images and masks.
        """
        images = []
        masks = []
        for img_id in img_ids:
            image, mask = self._load_image_and_mask(img_id)
            images.append(image)
            masks.append(mask)
        return self.preprocess(torch.stack(images), tv_tensors.Mask(torch.stack(masks)))


class UltrasoundTrainDataset(BaseUltrasoundDataset):
    def __init__(self, data_dir: str, annotations_file: str, image_size: tuple, sequence_length: int, truncated_bptt_steps: int, batch_size: int, train: bool = True):
        """
        Initialize the UltrasoundTrainDataset.

        Args:
            data_dir (str): Directory containing the data.
            annotations_file (str): Path to the annotations file.
            image_size (tuple): Size of the input image.
            sequence_length (int): Sequence length.
            truncated_bptt_steps (int): Truncated backpropagation through time steps.
            batch_size (int): Batch size.
            train (bool): Whether the dataset is for training. Default is True.
        """
        super().__init__(data_dir, annotations_file, image_size, sequence_length, batch_size)
        self.truncated_bptt_steps = truncated_bptt_steps
        self.sequence_length = sequence_length
        self.total_seq_len = sequence_length * truncated_bptt_steps
        self.train = train
        self._create_img_list()

    def _create_img_list(self):
        """
        Create a list of image sequences for training.

        This method generates sequences of image IDs for each video, considering the sequence length and truncated BPTT steps.
        """
        img_list = []
        for video_id, img_ids in self.video_id_to_img_ids.items():
            start = (
                random.randint(-(self.total_seq_len % len(img_ids)), 0) + len(img_ids) if self.train else 0
            )  # randint includes both ends
            reversed = True if self.train and random.random() > 0.5 else False
            for i in range(self.truncated_bptt_steps):
                end = start + self.sequence_length
                if end >= len(img_ids):
                    end -= len(img_ids)
                    sampled_ids = img_ids[start:] + img_ids[:end]
                else:
                    sampled_ids = img_ids[start:end]
                img_list.append((video_id, sampled_ids[::-1] if reversed else sampled_ids))
                start = end
        self.img_list = img_list

    def __len__(self) -> int:
        return len(self.video_id_to_img_ids) * self.truncated_bptt_steps

    def __getitem__(self, idx: int) -> tuple:
        video_id, img_ids = self.img_list[idx]
        images, masks = self._load_images_and_masks(img_ids)
        return images, {"masks": masks}


class UltrasoundTestDataset(BaseUltrasoundDataset):
    def __init__(self, data_dir: str, annotations_file: str, image_size: tuple, sequence_length: int, batch_size: int):
        """
        Initialize the UltrasoundTestDataset.

        Args:
            data_dir (str): Directory containing the data.
            annotations_file (str): Path to the annotations file.
            image_size (tuple): Size of the input image.
            sequence_length (int): Sequence length.
            batch_size (int): Batch size.
        """
        super().__init__(data_dir, annotations_file, image_size, sequence_length, batch_size)
        self.img_list = self._create_img_list()

    def _create_img_list(self) -> list:
        img_list = []
        for video_id, img_ids in self.video_id_to_img_ids.items():
            for i in range(0, len(img_ids), self.sequence_length):
                img_list.append((video_id, img_ids[i : min(len(img_ids), i + self.sequence_length)]))
        return img_list

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx: int) -> tuple:
        video_id, img_ids = self.img_list[idx]
        images, masks = self._load_images_and_masks(img_ids)
        return images, {"video_id": video_id, "image_ids": img_ids, "masks": masks}


class DistributedVideoSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, drop_last: bool = False) -> None:
        """
        Initialize the DistributedVideoSampler.

        Args:
            dataset (Dataset): Dataset to be sampled.
            num_replicas (Optional[int]): Number of replicas. Default is None.
            rank (Optional[int]): Rank of the current process. Default is None.
            shuffle (bool): Whether to shuffle the data. Default is True.
            seed (int): Random seed. Default is 0.
            drop_last (bool): Whether to drop the last incomplete batch. Default is False.
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.clips_per_rank = (len(self.dataset.video_id_to_img_ids) // self.num_replicas) * self.dataset.truncated_bptt_steps
        print(f"Last {len(self.dataset.video_id_to_img_ids) % self.num_replicas} videos will be dropped")

    def __iter__(self) -> Iterator[List[int]]:
        """
        Return an iterator over the indices of the dataset.

        This method generates indices for the current process based on the rank and number of replicas.
        """
        indices = list(range(len(self.dataset)))
        # subsample
        indices = indices[self.clips_per_rank * self.rank: self.clips_per_rank  * (self.rank + 1)]
        return iter(indices)
