import json

import torch
from torch.utils.data import Dataset

from collections import defaultdict
import os
import cv2
import numpy as np
import math

from utils import safe_load

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
    def __init__(self, json_folder, img_folder, trim_ends=False):
        json_paths = {}
        for root, folders, files in os.walk(json_folder):
            for f in files:
                if f.lower().endswith('.json'):
                    img_id = f.split(".")[0]
                    json_paths[img_id] = os.path.join(root, f)

        img_paths = {}
        for root, folders, files in os.walk(img_folder):
            for f in files:
                if f.lower().endswith('.jpg') or f.lower().endswith('.png'):
                    img_id = f.split(".")[0]
                    img_paths[img_id] = os.path.join(root, f)

        self.img_paths = img_paths
        self.json_paths = json_paths

        self.ids = list(set(json_paths.keys()) & set(img_paths.keys()))
        self.ids.sort()

        self.detailed_ids = []
        for image_key in self.ids:
            #with open(json_paths[image_key]) as f:
            #    d = json.load(f)
            d = safe_load.json_state(json_paths[image_key])
            if d is None:
                continue

            for i in xrange(len(d)):
                if 'lf' not in d[i]:
                    continue
                # if 'pred_full' not in d[i]:
                #     continue
                self.detailed_ids.append((image_key, i))


        print(len(self.ids))
        self.ids = self.detailed_ids
        self.trim_ends = trim_ends

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        image_key, line_idx = self.ids[idx]
        gt_json_path = self.json_paths[image_key]
        #with open(gt_json_path) as f:
        #    gt_json = json.load(f)
        gt_json = safe_load.json_state(gt_json_path)

        img_path = self.img_paths[image_key]

        positions = []
        positions_xy = []

        #if 'pred_full' in gt_json[line_idx]:
        if 'pred_full' in gt_json:
            pred_full = gt_json[line_idx]['pred_full']

            last_idx = 0
            for i in range(len(pred_full)):
                if pred_full[i] != "~":
                    last_idx = i+1

            percent = last_idx / float(len(pred_full))
            trim_idx = int(len(gt_json[line_idx]['lf']) * percent+1)

            trim_idx = max(2, trim_idx)
            gt_json[line_idx]['lf'] = gt_json[line_idx]['lf'][:trim_idx]

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

        img = cv2.imread(img_path).astype(np.float32)
        img = img.transpose()
        img = img / 128.0 - 1.0
        img = torch.from_numpy(img)

        result = {
            "img": img,
            "lf_xyrs": positions,
            "lf_xyxy": positions_xy
        }
        return result
