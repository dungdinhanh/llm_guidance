from typing import Callable, List, Optional, Union
# from hfai.datasets.base import BaseDataset, get_data_dir, register_dataset
import os
from pycocotools.coco import COCO
import numpy as np
from pathlib import Path
# from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data import DataLoader
from datasets.dataloader import *

DATA_DIR = None
DEFAULT_DATA_DIR = Path("/public_dataset/1/ffdataset")


def set_data_dir(path: Union[str, os.PathLike]) -> None:
    """
    设置数据集存放的主目录

    我们会优先使用通过 ``set_data_dir`` 设置的路径，如果没有则会去使用环境变量 ``HFAI_DATASETS_DIR`` 的值。
    两者都没有设置的情况下，使用默认目录 ``/public_dataset/1/ffdataset``。

    Args:
        path (str, os.PathLike): 数据集存放的主目录

    Examples:

        >>> hfai.datasets.set_data_dir("/your/data/dir")
        >>> hfai.datasets.get_data_dir()
        PosixPath('/your/data/dir')

    """
    global DATA_DIR
    DATA_DIR = Path(path).absolute()


def get_data_dir() -> Path:
    """
    返回当前数据集主目录

    Returns:
        data_dir (Path): 当前数据集主目录

    Examples:

        >>> hfai.datasets.set_data_dir("/your/data/dir")
        >>> hfai.datasets.get_data_dir()
        PosixPath('/your/data/dir')

    """
    global DATA_DIR

    # 1. set_data_dir() 设置的路径
    if DATA_DIR is not None:
        return DATA_DIR.absolute()

    # 2. 环境变量 HFAI_DATASETS_DIR 指定的路径
    env = os.getenv("HFAI_DATASETS_DIR")
    if env is not None:
        return Path(env)

    # 3. 默认路径
    return DEFAULT_DATA_DIR


class CocoCaptiononlyNCI(Dataset):
    def __init__(self, split, transform: Optional[Callable]=None, check_data: bool = True):
        data_dir = get_data_dir()
        self.data_dir = os.path.join(data_dir, "COCO")
        assert split in ["train", "val"]
        self.split = split
        self.annotation_path = os.path.join(self.data_dir, f"annotations/captions_{self.split}2017.json")
        self.coco = COCO(self.annotation_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, indices):
        n = len(indices)
        selected = np.random.randint(0, 5, (n,))
        annos = []
        for i in range(n):
            annos.append(self.read_anno(indices[i])[selected[i]]['caption'])
        return annos

    def __len__(self):
        return len(self.ids)

    def loader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self, *args, **kwargs)

    def read_anno(self, index: int) -> List[dict]:
        """
        读取指定索引下的注解信息。更多信息参考：https://cocodataset.org/#format-data

        Args:
            index (int): 指定的索引

        Returns:
            注解信息，返回一个包含若干字典组成的列表，每一个列表里包括 ``instance`` 和 ``contains``，例如：

            .. code-block:: python

                captions:
                    {'image_id': 444010,
                     'id': 104057,
                     'caption': 'A group of friends sitting down at a table sharing a meal.'}

                instances:
                    {'segmentation': ...,
                     'area': 3514.564,
                     'iscrowd': 0,
                     'image_id': 444010,
                     'bbox': [x_left, y_top, w, h],
                     'category_id': 44,
                     'id': 91863}

                keypoints:
                    {'segmentation': ...,
                     'num_keypoints': 11,
                     'area': 34800.5498,
                     'iscrowd': 0,
                     'keypoints': ...,
                     'image_id': 444010,
                     'bbox': [x_left, y_top, w, h],
                     'category_id': 1,
                     'id': 1200757}

                panoptic:
                    {"image_id": int,
                     "file_name": str,
                     "segments_info":
                        {
                        "id": int,
                        "category_id": int,
                        "area": int,
                        "bbox": [x,y,width,height],
                        "iscrowd": 0 or 1,
                        },
                     }
        """
        img_id = self.ids[index]
        ann_id = self.coco.getAnnIds(img_id)
        ann = self.coco.loadAnns(ann_id)
        return ann

import json
import pandas as pd
class CocoCaptionBBonlyNCI(Dataset):
    def __init__(self, path_csv:str):
        df = pd.read_csv(path_csv)
        self.annotations = list(df["text"])

    def __getitem__(self, indices):
        n = len(indices)
        # selected = np.random.randint(0, 5, (n,))
        annos = []
        for i in range(n):
            annos.append(self.annotations[indices[i]])
        return annos

    def __len__(self):
        return len(self.annotations)

    def loader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self, *args, **kwargs)


    
class CocoCaptionWOthers(CocoCaptiononlyNCI):
    def __init__(
            self,
            split: str,
            transform: Optional[Callable] = None,
            check_data: bool = True,
            # miniset: bool = False,
            # data_folder="data"
    ):
        super().__init__(split, transform, check_data)

    def __getitem__(self, indices):
        n = len(indices)
        selected = np.random.randint(0, 5, (n,))
        annos = []
        annos_others = []
        for i in range(n):
            annos_other = []
            anno_list = self.read_anno(indices[i])
            len_list_anno = len(anno_list)
            for j in range(len_list_anno):
                if j == selected[i]:
                    annos.append(anno_list[j]['caption'])
                else:
                    if len(annos_other) >= 4:
                        continue
                    annos_other.append(anno_list[j]['caption'])
            annos_others.append(annos_other)
        return list(zip(annos, annos_others))

def load_data_caption_hfai(
        *,
        split: str,
        batch_size: int,
):
    dataset = CocoCaptiononlyNCI(split)

    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True, drop_last=True)

    while True:
        yield from loader  # put all items of loader into list and concat all list infinitely

def load_data_caption_hfai_one_process(
        *,
        split: str,
        batch_size: int,
):
    dataset = CocoCaptiononlyNCI(split)

    # data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, pin_memory=True, drop_last=True)

    while True:
        yield from loader  # put all items of loader into list and concat all list infinitely

def load_data_caption_bb_hfai(
        *,
        split: str,
        batch_size: int,
        path,
):
    dataset = CocoCaptionBBonlyNCI(path)

    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True, drop_last=True)

    while True:
        yield from loader  # put all items of loader into list and concat all list infinitely

def load_data_5captions_hfai(
        *,
        split: str,
        batch_size: int,
):
    dataset = CocoCaptionWOthers(split)

    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True, drop_last=True)

    while True:
        yield from loader  # put all items of loader into list and concat all list infinitely