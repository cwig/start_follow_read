from torch.utils.data import Dataset
import json
import os
import cv2
import numpy as np
import torch
import random
from utils import safe_load

def collate(batch):
    return batch

class AlignmentDataset(Dataset):

    def __init__(self, set_list, data_range=None, ignore_json=False, resize_width=512):

        self.ignore_json = ignore_json

        self.resize_width = resize_width

        self.ids = set_list
        self.ids.sort()

        if data_range is not None:
            self.ids = random.sample(self.ids, data_range)

        print "Alignment Ids Count:", len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        gt_json_path, img_path = self.ids[idx]

        gt_json = []
        if not self.ignore_json:
            gt_json = safe_load.json_state(gt_json_path)
            if gt_json is None:
                return None

        org_img = cv2.imread(img_path)

        full_img = org_img.astype(np.float32)
        full_img = full_img.transpose([2,1,0])[None,...]
        full_img = torch.from_numpy(full_img)
        full_img = full_img / 128 - 1

        target_dim1 = self.resize_width
        s = target_dim1 / float(org_img.shape[1])
        target_dim0 = int(org_img.shape[0]/float(org_img.shape[1]) * target_dim1)

        img = cv2.resize(org_img,(target_dim1, target_dim0), interpolation = cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img = img.transpose([2,1,0])[None,...]
        img = torch.from_numpy(img)
        img = img / 128 - 1

        image_key = gt_json_path[:-len('.json')]

        return {
            "resized_img": img,
            "full_img": full_img,
            "resize_scale": 1.0/s,
            "gt_lines": [x['gt'] for x in gt_json],
            "img_key": image_key,
            "json_path": gt_json_path,
            "gt_json": gt_json
        }
