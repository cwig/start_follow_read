import json

import torch
from torch.utils.data import Dataset

from collections import defaultdict
import os
import cv2
import numpy as np
import math
import random

from utils import safe_load, augmentation

def collate(batch):
    return batch


def get_subdivide_pt(i, pred_full, lf):
    percent = (float(i)+0.5) / float(len(pred_full))
    lf_percent = (len(lf)-1) * percent

    lf_idx = int(np.floor(lf_percent))
    step_percent = lf_percent - lf_idx

    x0 = lf[lf_idx]['cx']
    y0 = lf[lf_idx]['cy']
    x1 = lf[lf_idx+1]['cx']
    y1 = lf[lf_idx+1]['cy']

    x = x0 * step_percent + x1 * (1.0 - step_percent)
    y = y0 * step_percent + y1 * (1.0 - step_percent)

    return x, y

class LfDataset(Dataset):
    def __init__(self, set_list, random_subset_size=None, augmentation=False):
        self.augmentation = augmentation

        self.ids = set_list
        self.ids.sort()

        self.detailed_ids = []
        for ids_idx, paths in enumerate(self.ids):
            json_path, img_path = paths

            d = safe_load.json_state(json_path)
            if d is None:
                continue

            for i in xrange(len(d)):
                if 'lf' not in d[i]:
                    continue
                self.detailed_ids.append((ids_idx, i))

        if random_subset_size is not None:
            self.detailed_ids = random.sample(self.detailed_ids, min(len(self.ids), random_subset_size))

        print(len(self.detailed_ids))

    def __len__(self):
        return len(self.detailed_ids)

    def __getitem__(self, idx):

        ids_idx, line_idx = self.detailed_ids[idx]
        gt_json_path, img_path = self.ids[ids_idx]
        gt_json = safe_load.json_state(gt_json_path)

        positions = []
        positions_xy = []


        if 'lf' not in gt_json[line_idx]:
            return None

        for step in gt_json[line_idx]['lf']:
            x0 = step['x0']
            x1 = step['x1']
            y0 = step['y0']
            y1 = step['y1']

            positions_xy.append((torch.Tensor([[x1,x0],[y1,y0]])))

            dx = x0-x1
            dy = y0-y1

            d = math.sqrt(dx**2 + dy**2)

            mx = (x0+x1)/2.0
            my = (y0+y1)/2.0

            #Not sure if this is right...
            theta = -math.atan2(dx, -dy)

            positions.append(torch.Tensor([mx, my, theta, d/2, 1.0]))

        img = cv2.imread(img_path)
        if self.augmentation:
            img = augmentation.apply_random_color_rotation(img)
            img = augmentation.apply_tensmeyer_brightness(img)



        img = img.astype(np.float32)
        img = img.transpose()
        img = img / 128.0 - 1.0
        img = torch.from_numpy(img)

        gt = gt_json[line_idx]['gt']

        result = {
            "img": img,
            "lf_xyrs": positions,
            "lf_xyxy": positions_xy,
            "gt": gt
        }
        return result
