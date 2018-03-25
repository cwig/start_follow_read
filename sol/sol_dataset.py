import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np
from utils import safe_load
import math
import random

def collate(batch):

    batch_size = len(batch)
    imgs = []
    label_sizes = []
    for b in batch:
        if b is None:
            continue
        imgs.append(b["img"])
        if b['sol_gt'] is None:
            label_sizes.append(0)
        else:
            label_sizes.append(b['sol_gt'].size(1))

    if len(imgs) == 0:
        return None
    batch_size = len(imgs)

    largest_label = max(label_sizes)

    labels = None
    if largest_label != 0:
        labels = torch.zeros(batch_size, largest_label, 4)
        for i, b in enumerate(batch):
            if label_sizes[i] == 0:
                continue
            labels[i, :label_sizes[i]] = b['sol_gt']

    imgs = torch.cat(imgs)

    return {
        'sol_gt': labels,
        'img': imgs,
        "label_sizes": label_sizes
    }

# CNT = 0
class SolDataset(Dataset):
    def __init__(self, json_folder, img_folder, rescale_range=None, transform=None, random_subset_size=None):

        json_paths = {}
        for root, folders, files in os.walk(json_folder):
            for f in files:
                if f.lower().endswith('.json'):
                    img_id = f.split(".")[0]
                    json_paths[img_id] = os.path.join(root, f)

        img_paths = {}
        for root, folders, files in os.walk(img_folder):
            for f in files:
                if f.lower().endswith('.jpg'):
                    img_id = f.split(".")[0]
                    img_paths[img_id] = os.path.join(root, f)

        self.rescale_range = rescale_range
        self.img_paths = img_paths
        self.json_paths = json_paths

        self.ids = list(set(json_paths.keys()) & set(img_paths.keys()))
        self.ids.sort()

        new_ids = []
        for image_key in self.ids:

            gt_json_path = self.json_paths[image_key]
            gt_json = safe_load.json_state(gt_json_path)
            if gt_json is None:
                continue
            failed = False
            for j, gt_item in enumerate(gt_json):
                if 'sol' not in gt_item:
                    failed = True
                    break

            if failed:
                continue
            new_ids.append(image_key)

        self.ids = new_ids

        if random_subset_size is not None:
            self.ids = random.sample(self.ids, min(random_subset_size, len(self.ids)))
        print "SOL Ids Count:", len(self.ids)
        self.transform = transform


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        image_key = self.ids[idx]
        gt_json_path = self.json_paths[image_key]

        gt_json = safe_load.json_state(gt_json_path)
        if gt_json is None:
            return None

        img_path = self.img_paths[image_key]

        org_img = cv2.imread(img_path)
        target_dim1 = int(np.random.uniform(self.rescale_range[0], self.rescale_range[1]))

        s = target_dim1 / float(org_img.shape[1])
        target_dim0 = int(org_img.shape[0]/float(org_img.shape[1]) * target_dim1)
        org_img = cv2.resize(org_img,(target_dim1, target_dim0), interpolation = cv2.INTER_CUBIC)

        gt = np.zeros((1,len(gt_json), 4), dtype=np.float32)

        for j, gt_item in enumerate(gt_json):
            if 'sol' not in gt_item:
                continue

            x0 = gt_item['sol']['x0']
            x1 = gt_item['sol']['x1']
            y0 = gt_item['sol']['y0']
            y1 = gt_item['sol']['y1']

            gt[:,j,0] = x0 * s
            gt[:,j,1] = y0 * s
            gt[:,j,2] = x1 * s
            gt[:,j,3] = y1 * s

        if self.transform is not None:
            out = self.transform({
                "img": org_img,
                "sol_gt": gt
            })
            org_img = out['img']
            gt = out['sol_gt']

        img = org_img.transpose([2,1,0])[None,...]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = img / 128.0 - 1.0

        if gt.shape[1] == 0:
            gt = None
        else:
            gt = torch.from_numpy(gt)

        return {
            "img": img,
            "sol_gt": gt
        }
