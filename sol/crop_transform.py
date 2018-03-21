
from sol import crop_utils
import numpy as np

class CropTransform(object):
    def __init__(self, crop_params):
        crop_size = crop_params['crop_size']
        self.random_crop_params = crop_params
        self.pad_params =  ((crop_size,crop_size),(crop_size,crop_size),(0,0))

    def __call__(self, sample):
        org_img = sample['img']
        gt = sample['sol_gt']

        org_img = np.pad(org_img, self.pad_params, 'mean')

        gt[:,:,0] = gt[:,:,0] + self.pad_params[0][0]
        gt[:,:,1] = gt[:,:,1] + self.pad_params[1][0]

        gt[:,:,2] = gt[:,:,2] + self.pad_params[0][0]
        gt[:,:,3] = gt[:,:,3] + self.pad_params[1][0]

        crop_params, org_img, gt_match = crop_utils.generate_random_crop(org_img, gt, self.random_crop_params)

        gt = gt[gt_match][None,...]
        gt[...,0] = gt[...,0] - crop_params['dim1'][0]
        gt[...,1] = gt[...,1] - crop_params['dim0'][0]

        gt[...,2] = gt[...,2] - crop_params['dim1'][0]
        gt[...,3] = gt[...,3] - crop_params['dim0'][0]

        return {
            "img": org_img,
            "sol_gt": gt
        }
