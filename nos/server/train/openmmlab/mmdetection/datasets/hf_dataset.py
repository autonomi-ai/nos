import copy
import logging

import numpy as np
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS
from mmengine.dataset import force_full_init

from datasets import load_dataset


logger = logging.getLogger(__name__)


@DATASETS.register_module()
class HuggingfaceDataset(BaseDetDataset):

    METAINFO = {
        "classes": ("person", "bicycle", "car", "motorcycle"),
        "palette": [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = load_dataset("cppe-5", split="train")

    @force_full_init
    def __len__(self):
        return len(self.dataset)

    def load_data_list(self):
        self.serialize_data = False
        return []

    def filter_data(self):
        return []

    def load_annotations(self, ann_file):
        data_infos = []
        for idx in range(len(self.dataset)):
            datum = self.dataset[idx]
            data_infos.append(
                {
                    "filename": datum["image_id"],
                    "width": datum["width"],
                    "height": datum["height"],
                    "ann": {
                        "bboxes": np.array(datum["objects"]["bbox"]).astype(np.float32),
                        "labels": np.array(datum["objects"]["category"]).astype(np.int64),
                    },
                }
            )
        return data_infos

    def __getitem__(self, idx):
        # See LoadImageFromFile and LoadAnnotations
        # https://github.com/open-mmlab/mmcv/blob/ee93530acc675231014b92a58fd6e4a59e27cc13/mmcv/transforms/loading.py#L135
        datum = self.dataset[idx]
        logger.debug(f"Loading datum [idx={idx}]: {datum['image_id']}")
        img = np.asarray(datum["image"].convert("RGB"))
        data = {
            "img_id": datum["image_id"],
            "img_path": datum["image_id"],
            "img": img,
            "img_shape": img.shape[:2],
            "ori_shape": img.shape[:2],
            "width": datum["width"],
            "height": datum["height"],
            "gt_bboxes": np.array(datum["objects"]["bbox"]).astype(np.float32),
            "gt_bboxes_labels": np.array(datum["objects"]["category"]).astype(np.int64),
            "gt_ignore_flags": [False] * len(datum["objects"]["bbox"]),
        }
        results = copy.deepcopy(self.pipeline(data))
        return results
