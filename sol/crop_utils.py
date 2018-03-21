import cv2
import numpy as np
import sys

def perform_crop(img, crop):
    cs = crop['crop_size']
    cropped_gt_img = img[crop['dim0'][0]:crop['dim0'][1], crop['dim1'][0]:crop['dim1'][1]]
    scaled_gt_img = cv2.resize(cropped_gt_img, (cs, cs), interpolation = cv2.INTER_CUBIC)
    return scaled_gt_img


def generate_random_crop(img, gt, params):

    contains_label = np.random.random() < params['prob_label']
    cs = params['crop_size']

    cnt = 0
    while True:

        dim0 = np.random.randint(0,img.shape[0]-cs)
        dim1 = np.random.randint(0,img.shape[1]-cs)

        crop = {
            "dim0": [dim0, dim0+cs],
            "dim1": [dim1, dim1+cs],
            "crop_size": cs
        }

        #TODO: this only works for the center points
        gt_match = np.zeros_like(gt[...,0:2])
        gt_match[...,0][gt[...,0] < dim1] = 1
        gt_match[...,0][gt[...,0] > dim1+cs] = 1

        gt_match[...,1][gt[...,1] < dim0] = 1
        gt_match[...,1][gt[...,1] > dim0+cs] = 1

        gt_match = 1-gt_match
        gt_match = np.logical_and(gt_match[...,0], gt_match[...,1])

        if gt_match.sum() > 0 and contains_label or cnt > 100:
            cropped_gt_img = perform_crop(img, crop)
            return crop, cropped_gt_img, np.where(gt_match != 0)

        if gt_match.sum() == 0 and not contains_label:
            cropped_gt_img = perform_crop(img, crop)
            return crop, cropped_gt_img, np.where(gt_match != 0)

        cnt += 1
