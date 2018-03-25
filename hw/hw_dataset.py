import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np
import math

import grid_distortion

from utils import string_utils, safe_load

import random
PADDING_CONSTANT = 0

def collate(batch):
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0]
    dim1 = max([b['line_img'].shape[1] for b in batch])
    dim2 = batch[0]['line_img'].shape[2]

    all_labels = []
    label_lengths = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in xrange(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,:b_img.shape[1],:] = b_img

        l = batch[i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))

    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0,3,1,2])
    line_imgs = torch.from_numpy(line_imgs)
    labels = torch.from_numpy(all_labels.astype(np.int32))
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    return {
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch]
    }

class HwDataset(Dataset):
    def __init__(self, json_folder, img_folder, char_to_idx, augmentation=False, img_height=32, random_subset_size=None):
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

        self.img_paths = img_paths
        self.json_paths = json_paths
        self.img_height = img_height

        self.ids = list(set(json_paths.keys()) & set(img_paths.keys()))
        self.ids.sort()

        self.detailed_ids = []
        for image_key in self.ids:
            d = safe_load.json_state(json_paths[image_key])
            if d is None:
                continue
            for i in xrange(len(d)):

                if 'hw_path' not in d[i]:
                    continue
                self.detailed_ids.append((image_key, i))

        if random_subset_size is not None:
            self.detailed_ids = random.sample(self.detailed_ids, min(random_subset_size, len(self.detailed_ids)))
        print len(self.detailed_ids)

        self.ids = self.detailed_ids
        self.char_to_idx = char_to_idx
        self.json_folder = json_folder
        self.augmentation = augmentation
        self.warning=False



    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_key, line_idx = self.ids[idx]

        gt_json_path = self.json_paths[image_key]
        gt_json = safe_load.json_state(gt_json_path)
        if gt_json is None:
            return None

        if 'hw_path' not in gt_json[line_idx]:
            return None

        hw_path = gt_json[line_idx]['hw_path']

        hw_path = hw_path.split("/")[-1:]
        hw_path = "/".join(hw_path)

        hw_folder = os.path.dirname(gt_json_path)

        img = cv2.imread(os.path.join(hw_folder, hw_path))

        if img is None:
            return None

        if img.shape[0] != self.img_height:
            if img.shape[0] < self.img_height and not self.warning:
                self.warning = True
                print "WARNING: upsampling image to fit size"
            percent = float(self.img_height) / img.shape[0]
            img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)

        if img is None:
            return None

        img = img.astype(np.float32)

        if self.augmentation:
            img = grid_distortion.warp_image(img)

        img = img / 128.0 - 1.0

        gt = gt_json[line_idx]['gt']
        gt_label = string_utils.str2label_single(gt, self.char_to_idx)


        return {
            "line_img": img,
            "gt": gt,
            "gt_label": gt_label
        }
