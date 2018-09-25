import sys
import os
from os import path, listdir
from os.path import join, isfile
from collections import defaultdict
import xml.etree.ElementTree as ET
import re
import json
import numpy as np
import torch
from torch.autograd import Variable
from e2e.e2e_model import E2EModel
from e2e import visualization
from utils.continuous_state import init_model
from hw import grid_distortion
from utils import PAGE_xml

import cv2
import codecs

import yaml

from e2e import e2e_postprocessing

import pyclipper
from copy import deepcopy

import argparse

def log_softmax(hw):
    line_data = Variable(torch.from_numpy(hw), requires_grad=False)
    softmax_out = torch.nn.functional.log_softmax(line_data, -1).data.numpy()
    return hw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('npz_folder')
    parser.add_argument('--in_xml_folder')
    parser.add_argument('--out_xml_folder')
    parser.add_argument('--lm', action='store_true')
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--roi', action='store_true')
    args = parser.parse_args()


    config_path = args.config_path
    npz_folder = args.npz_folder
    in_xml_folder = args.in_xml_folder
    out_xml_folder = args.out_xml_folder

    in_xml_files = {}
    if in_xml_folder and out_xml_folder:
        for root, folders, files in os.walk(in_xml_folder):
            for f in files:
                if f.endswith(".xml"):
                    basename = os.path.basename(f).replace(".xml", "")
                    in_xml_files[basename] = os.path.join(root, f)



    use_lm = args.lm
    use_aug = args.aug
    use_roi = args.roi

    if use_lm:
        from utils import lm_decoder

    with open(config_path) as f:
        config = yaml.load(f)

    npz_paths = []
    for root, folder, files in os.walk(npz_folder):
        for f in files:
            if f.lower().endswith(".npz"):
                npz_paths.append(os.path.join(root, f))

    char_set_path = config['network']['hw']['char_set_path']

    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k,v in char_set['idx_to_char'].iteritems():
        idx_to_char[int(k)] = v

    if use_aug:
        model_mode = "best_overall"
        _,_, hw = init_model(config, hw_dir=model_mode, only_load="hw")
        dtype = torch.cuda.FloatTensor
        hw.eval()

    if use_lm:
        lm_params = config['network']['lm']
        print "Loading LM"
        decoder = lm_decoder.LMDecoder(idx_to_char, lm_params)
        print "Done Loading LM"

        print "Accumulating stats for LM"
        for npz_path in sorted(npz_paths):
            out = np.load(npz_path)
            out = dict(out)
            for o in out['hw']:
                o = log_softmax(o)
                decoder.add_stats(o)
        print "Done accumulating stats for LM"
    else:
        print "Skip Loading LM"

    for npz_path in sorted(npz_paths):

        out = np.load(npz_path)
        out = dict(out)

        image_path = str(out['image_path'])
        print image_path
        org_img = cv2.imread(image_path)

        # Postprocessing Steps
        out['idx'] = np.arange(out['sol'].shape[0])
        out = e2e_postprocessing.trim_ends(out)
        e2e_postprocessing.filter_on_pick(out, e2e_postprocessing.select_non_empty_string(out))
        out = e2e_postprocessing.postprocess(out,
            sol_threshold=config['post_processing']['sol_threshold'],
            lf_nms_params={
                "overlap_range": config['post_processing']['lf_nms_range'],
                "overlap_threshold": config['post_processing']['lf_nms_threshold']
            }
            # },
            # lf_nms_2_params={
            #     "overlap_threshold": 0.5
            # }
        )
        order = e2e_postprocessing.read_order(out)
        e2e_postprocessing.filter_on_pick(out, order)

        # Decoding network output
        output_strings = []
        if use_aug:
            number_of_iterations = 20
            for line_img in out['line_imgs']:
                batch = []
                for i in range(number_of_iterations):
                    warped_image = grid_distortion.warp_image(line_img)
                    batch.append(warped_image)

                batch = np.array(batch)
                batch = Variable(torch.from_numpy(batch), requires_grad=False, volatile=True).cuda()
                batch = batch/128.0 - 1.0
                batch = batch.transpose(2,3)
                batch = batch.transpose(1,2)
                hw_out = hw(batch)
                hw_out = hw_out.transpose(0,1)
                hw_out=hw_out.data.cpu().numpy()

                if use_lm:
                    decoded_hw = []
                    for line in hw_out:
                        log_softmax_line = log_softmax(line)
                        lm_output = decoder.decode(log_softmax_line)[0]
                        decoded_hw.append(lm_output)
                else:
                    decoded_hw, decoded_raw_hw = e2e_postprocessing.decode_handwriting({"hw":hw_out}, idx_to_char)

                cnt_d = defaultdict(list)
                for i,d in enumerate(decoded_hw):
                    cnt_d[d].append(i)

                cnt_d = dict(cnt_d)
                sorted_list = list(sorted(cnt_d.iteritems(), key=lambda x:len(x[1])))

                best_idx = sorted_list[-1][1][0]
                output_strings.append(decoded_hw[best_idx])

        else:
            if use_lm:
                for line in out['hw']:
                    log_softmax_line = log_softmax(line)
                    lm_output = decoder.decode(log_softmax_line)[0]
                    output_strings.append(lm_output)
            else:
                output_strings, decoded_raw_hw = e2e_postprocessing.decode_handwriting(out, idx_to_char)

        draw_img = visualization.draw_output(out, org_img)
        cv2.imwrite(npz_path+".png", draw_img)

        # Save results
        label_string = "_"
        if use_lm:
            label_string += "lm_"
        if use_aug:
            label_string += "aug_"
        filepath = npz_path + label_string + ".txt"

        with codecs.open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_strings))

        key = os.path.basename(image_path)[:-len(".jpg")]
        if in_xml_folder:
            if use_roi:

                key,region_id = key.split("_",1)
                region_id = region_id.split(".")[0]

                if key in in_xml_files:
                    in_xml_file = in_xml_files[key]
                    out_xml_file = os.path.join(out_xml_folder, os.path.basename(in_xml_file))
                    PAGE_xml.create_output_xml_roi(in_xml_file, out, output_strings, out_xml_file, region_id)
                    in_xml_files[key] = out_xml_file #after first, add to current xml
                else:
                    print "Couldn't find xml file for ", key
            else:
                if key in in_xml_files:
                    in_xml_file = in_xml_files[key]
                    out_xml_file = os.path.join(out_xml_folder, os.path.basename(in_xml_file))
                    PAGE_xml.create_output_xml(in_xml_file, out, output_strings, out_xml_file)
                else:
                    print "Couldn't find xml file for ", key




if __name__ == "__main__":
    main()
