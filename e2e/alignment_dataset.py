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

    def __init__(self, json_folder, img_folder, data_range=None, ignore_json=False, resize_width=512):

        json_paths = {}
        for root, folders, files in os.walk(json_folder):
            for f in files:
                if f.lower().endswith('.json'):
                    img_id = f.split(".")[0]
                    json_paths[img_id] = os.path.join(root, f)

        img_paths = {}
        all_imgs = []
        for root, folders, files in os.walk(img_folder):
            for f in files:
                if f.lower().endswith('.jpg'):
                    img_id = f.split(".")[0]
                    all_imgs.append(f)
                    img_paths[img_id] = os.path.join(root, f)


        self.img_paths = img_paths
        self.json_paths = json_paths
        self.ignore_json = ignore_json

        self.resize_width = resize_width

        if self.ignore_json:
            self.ids = list(img_paths.keys())
        else:
            self.ids = list(set(json_paths.keys()) & set(img_paths.keys()))
        self.ids.sort()

        if data_range is not None:
            self.ids = random.sample(self.ids, data_range)

        print "Alignment Ids Count:", len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        image_key = self.ids[idx]
        # with open(gt_json_path) as f:
        #     gt_json = json.load(f)

        gt_json_path = None
        gt_json = []
        if not self.ignore_json:
            gt_json_path = self.json_paths[image_key]
            gt_json = safe_load.json_state(gt_json_path)
            if gt_json is None:
                return None


        img_path = self.img_paths[image_key]

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

        return {
            "resized_img": img,
            "full_img": full_img,
            "resize_scale": 1.0/s,
            "gt_lines": [x['gt'] for x in gt_json],
            "img_key": image_key,
            "json_path": gt_json_path,
            "gt_json": gt_json
        }
