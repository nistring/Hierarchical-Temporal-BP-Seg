import random
from typing import Optional, List, Iterator
import math

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision import tv_tensors
from pycocotools.coco import COCO

category_match = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 6, 9: 7, 10: 8}

class BaseUltrasoundDataset(Dataset):
    def __init__(self, data_dir: str, annotations_file: str, image_size: tuple, sequence_length: int, batch_size: int):
        self.data_dir = data_dir
        self.coco = COCO(annotations_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.num_classes = 10
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.video_id_to_img_ids = self._group_images_by_video()
        self.preprocess = v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def _group_images_by_video(self) -> dict:
        video_groups = {}
        for img_id in self.ids:
            video_id = self.coco.loadImgs(img_id)[0]["video_id"]
            video_groups.setdefault(video_id, []).append(img_id)
        return video_groups

    def _load_image_and_mask(self, img_id: int) -> tuple:
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]["file_name"]

        image = decode_image(str(self.data_dir / path), mode="GRAY")
        mask = torch.zeros(image.shape[1], image.shape[2], dtype=torch.uint8)

        for ann in annotations:
            ann_mask = torch.tensor(self.coco.annToMask(ann), dtype=torch.bool)
            mask[ann_mask] = category_match[ann["category_id"]]

        return image, mask

    def _load_images_and_masks(self, img_ids: list) -> tuple:
        images, masks = zip(*[self._load_image_and_mask(img_id) for img_id in img_ids])
        return self.preprocess(torch.stack(images), tv_tensors.Mask(torch.stack(masks)))


class UltrasoundTrainDataset(BaseUltrasoundDataset):
    def __init__(self, data_dir: str, annotations_file: str, image_size: tuple, sequence_length: int, 
                 truncated_bptt_steps: int, batch_size: int, train: bool = True):
        super().__init__(data_dir, annotations_file, image_size, sequence_length, batch_size)
        self.truncated_bptt_steps = truncated_bptt_steps
        self.total_seq_len = sequence_length * truncated_bptt_steps
        self.train = train
        self._create_img_list()

    def _create_img_list(self):
        img_list = []
        for video_id, img_ids in self.video_id_to_img_ids.items():
            start = random.randint(-(self.total_seq_len % len(img_ids)), 0) + len(img_ids) if self.train else 0
            if self.train and random.random() > 0.5:
                bi_img_ids = img_ids[::-1] + img_ids
            else:
                bi_img_ids = img_ids + img_ids[::-1]

            for i in range(self.truncated_bptt_steps):
                end = start + self.sequence_length
                if end >= len(bi_img_ids):
                    end -= len(bi_img_ids)
                    sampled_ids = bi_img_ids[start:] + bi_img_ids[:end]
                else:
                    sampled_ids = bi_img_ids[start:end]
                img_list.append((video_id, sampled_ids))
                start = end
        self.img_list = img_list

    def __len__(self) -> int:
        return len(self.video_id_to_img_ids) * self.truncated_bptt_steps

    def __getitem__(self, idx: int) -> tuple:
        video_id, img_ids = self.img_list[idx]
        images, masks = self._load_images_and_masks(img_ids)
        return images, {"masks": masks, "idx": idx}


class UltrasoundTestDataset(BaseUltrasoundDataset):
    def __init__(self, data_dir: str, annotations_file: str, image_size: tuple, sequence_length: int, batch_size: int):
        super().__init__(data_dir, annotations_file, image_size, sequence_length, batch_size)
        self.img_list = self._create_img_list()

    def _create_img_list(self) -> list:
        img_list = []
        for video_id, img_ids in self.video_id_to_img_ids.items():
            for i in range(0, len(img_ids), self.sequence_length):
                img_list.append((video_id, img_ids[i:i + self.sequence_length]))
        return img_list

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx: int) -> tuple:
        video_id, img_ids = self.img_list[idx]
        images, masks = self._load_images_and_masks(img_ids)
        return images, {"video_id": video_id, "image_ids": img_ids, "masks": masks}


class DistributedVideoSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None, 
                 shuffle: bool = False, seed: int = 0, drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        num_videos = len(self.dataset.video_id_to_img_ids)
        
        if drop_last:
            self.videos_per_rank = num_videos // (self.num_replicas * self.dataset.batch_size) * self.dataset.batch_size
        else:
            self.videos_per_rank = math.ceil(num_videos / (self.num_replicas * self.dataset.batch_size)) * self.dataset.batch_size
            
        self.padding_size = self.num_replicas * self.videos_per_rank - num_videos
        self.num_samples = self.videos_per_rank * self.dataset.truncated_bptt_steps

    def __iter__(self) -> Iterator[List[int]]:
        indices = torch.arange(len(self.dataset)).reshape(-1, self.dataset.truncated_bptt_steps)
        
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = indices[torch.randperm(len(indices), generator=g).tolist()]
            
        if not self.drop_last:
            indices = torch.cat((indices, indices[:self.padding_size])) 

        indices = indices[self.videos_per_rank * self.rank: self.videos_per_rank * (self.rank + 1)]
        indices = indices.reshape(-1, self.dataset.batch_size, self.dataset.truncated_bptt_steps).transpose(1, 2).reshape(-1)
        return iter(indices)