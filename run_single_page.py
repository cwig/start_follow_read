import sys
import torch

from utils.continuous_state import init_model
from e2e import e2e_model
from e2e.e2e_model import E2EModel
from e2e import e2e_postprocessing, visualization

import torch
from torch import nn
from torch.autograd import Variable

import json
import cv2
import numpy as np
import os
import codecs
import yaml
import sys



def main():
    config_path = sys.argv[1]
    image_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(config_path) as f:
        config = yaml.load(f)

    char_set_path = config['network']['hw']['char_set_path']

    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k,v in char_set['idx_to_char'].iteritems():
        idx_to_char[int(k)] = v

    model_mode = "pretrain"
    sol, lf, hw = init_model(config, sol_dir=model_mode, lf_dir=model_mode, hw_dir=model_mode)
    e2e = E2EModel(sol, lf, hw)
    e2e.eval()
    dtype = torch.cuda.FloatTensor

    org_img = cv2.imread(image_path)

    target_dim1 = 512
    s = target_dim1 / float(org_img.shape[1])

    target_dim0 = int(org_img.shape[0] * s)
    target_dim1 = int(org_img.shape[1] * s)

    full_img = org_img.astype(np.float32)
    full_img = full_img.transpose([2,1,0])[None,...]
    full_img = torch.from_numpy(full_img)
    full_img = full_img / 128 - 1

    img = cv2.resize(org_img,(target_dim1, target_dim0), interpolation = cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = img.transpose([2,1,0])[None,...]
    img = torch.from_numpy(img)
    img = img / 128 - 1

    out = e2e.forward({
        "resized_img": img,
        "full_img": full_img,
        "resize_scale": 1.0/s
    }, use_full_img=True)

    out = e2e_postprocessing.results_to_numpy(out)
    decoded_hw, decoded_raw_hw = e2e_postprocessing.decode_handwriting(out, idx_to_char)
    out['idx'] = np.arange(out['sol'].shape[0])

    out = e2e_postprocessing.trim_ends(out)

    e2e_postprocessing.filter_on_pick(out, e2e_postprocessing.select_non_empty_string(out))

    out = e2e_postprocessing.postprocess(out,
        sol_threshold=0.1,
        lf_nms_params={
            "overlap_range": [0,6],
            "overlap_threshold": 0.5
    })

    order = e2e_postprocessing.read_order(out)
    e2e_postprocessing.filter_on_pick(out, order)

    preds = ["{}: ".format(cnt) + decoded_hw[i] for cnt, i in enumerate(out['idx'])]
    raw_preds = ["{}: ".format(cnt) + decoded_raw_hw[i] for cnt, i in enumerate(out['idx'])]

    pred = "\n".join(preds)
    raw_pred = "\n".join(raw_preds)

    draw_img = visualization.draw_output(out, org_img)

    cv2.imwrite(output_path, draw_img)

if __name__ == "__main__":
    main()
