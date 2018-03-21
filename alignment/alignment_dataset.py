from torch.utils.data import Dataset
import json
import os
import cv2
import numpy as np
import torch

def collate(batch):
    return batch

class AlignmentDataset(Dataset):

    def __init__(self, gt_json_path, img_folder, char_set):

        with open(gt_json_path) as f:
            gt_labels = json.load(f)

        img_paths = {}
        for root, folders, files in os.walk(img_folder):
            for f in files:
                if f.lower().endswith('.jpg'):
                    img_id = f.split(".")[0]
                    img_paths[img_id] = os.path.join(root, f)


        gt_lines = {}
        for v in gt_labels:
            key = v['image_path'].split("/")[-1].split(".")[0]
            gt_lines[key] = v['gt_lines']


        self.img_ids = list(set(img_paths.keys()) & set(gt_lines.keys()))
        self.img_ids.sort()

        self.gt_lines = gt_lines
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_key = self.img_ids[idx]
        img_path = self.img_paths[img_key]
        org_img = cv2.imread(img_path)

        if org_img is None:
            return None

        full_img = org_img.astype(np.float32)
        full_img = full_img.transpose([2,1,0])[None,...]
        full_img = torch.from_numpy(full_img)
        full_img = full_img / 128 - 1

        target_dim1 = 512
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
            "gt_lines": self.gt_lines[img_key],
            "img_key": img_key
        }
