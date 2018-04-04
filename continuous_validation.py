import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


from sol.start_of_line_finder import StartOfLineFinder
from lf.line_follower import LineFollower
from hw import cnn_lstm
from utils import safe_load, error_rates

import numpy as np
import cv2
import os
import sys
import json
import time
import yaml
import operator


from e2e import e2e_model
from e2e.e2e_model import E2EModel

from e2e import alignment_dataset, e2e_postprocessing, visualization
from e2e.alignment_dataset import AlignmentDataset

from utils.continuous_state import init_model

import itertools
from collections import defaultdict
from utils import error_rates
import copy
from copy import deepcopy

from e2e import validation_utils
from utils.dataset_parse import load_file_list

def alignment_step(config, dataset_lookup=None, model_mode='best_validation', percent_range=None):

    set_list = load_file_list(config['training'][dataset_lookup])

    if percent_range is not None:
        start = int(len(set_list) * percent_range[0])
        end = int(len(set_list) * percent_range[1])
        set_list = set_list[start:end]

    dataset = AlignmentDataset(set_list, None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=alignment_dataset.collate)

    char_set_path = config['network']['hw']['char_set_path']

    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k,v in char_set['idx_to_char'].iteritems():
        idx_to_char[int(k)] = v

    sol, lf, hw = init_model(config, sol_dir=model_mode, lf_dir=model_mode, hw_dir=model_mode)

    e2e = E2EModel(sol, lf, hw)
    dtype = torch.cuda.FloatTensor
    e2e.eval()

    post_processing_config = config['training']['alignment']['validation_post_processing']
    sol_thresholds = post_processing_config['sol_thresholds']
    sol_thresholds_idx = range(len(sol_thresholds))

    lf_nms_ranges = post_processing_config['lf_nms_ranges']
    lf_nms_ranges_idx = range(len(lf_nms_ranges))

    lf_nms_thresholds = post_processing_config['lf_nms_thresholds']
    lf_nms_thresholds_idx = range(len(lf_nms_thresholds))

    results = defaultdict(list)
    aligned_results = []
    best_ever_results = []

    prev_time = time.time()
    cnt = 0
    a = 0
    for x in dataloader:
        sys.stdout.flush()
        a+=1

        if a%100 == 0:
            print a, np.mean(aligned_results)


        x = x[0]
        if x is None:
            print "Skipping alignment because it returned None"
            continue

        img = x['resized_img'].numpy()[0,...].transpose([2,1,0])
        img = ((img+1)*128).astype(np.uint8)

        full_img = x['full_img'].numpy()[0,...].transpose([2,1,0])
        full_img = ((full_img+1)*128).astype(np.uint8)

        gt_lines = x['gt_lines']
        gt = "\n".join(gt_lines)

        out_original = e2e(x)
        if out_original is None:
            #TODO: not a good way to handle this, but fine for now
            print "Possible Error: Skipping alignment on image"
            continue

        out_original = e2e_postprocessing.results_to_numpy(out_original)
        out_original['idx'] = np.arange(out_original['sol'].shape[0])
        e2e_postprocessing.trim_ends(out_original)
        decoded_hw, decoded_raw_hw = e2e_postprocessing.decode_handwriting(out_original, idx_to_char)
        pick, costs = e2e_postprocessing.align_to_gt_lines(decoded_hw, gt_lines)


        best_ever_pred_lines, improved_idxs = validation_utils.update_ideal_results(pick, costs, decoded_hw, x['gt_json'])
        validation_utils.save_improved_idxs(improved_idxs, decoded_hw,
                                            decoded_raw_hw, out_original,
                                            x, config['training'][dataset_lookup]['json_folder'])

        best_ever_pred_lines = "\n".join(best_ever_pred_lines)
        error = error_rates.cer(gt, best_ever_pred_lines)
        best_ever_results.append(error)

        aligned_pred_lines = [decoded_hw[i] for i in pick]
        aligned_pred_lines = "\n".join(aligned_pred_lines)
        error = error_rates.cer(gt, aligned_pred_lines)
        aligned_results.append(error)


        if dataset_lookup == "validation_set":
            # We only care about the hyperparameter postprocessing seach for the validation set
            for key in itertools.product(sol_thresholds_idx, lf_nms_ranges_idx, lf_nms_thresholds_idx):
                i,j,k = key
                sol_threshold = sol_thresholds[i]
                lf_nms_range = lf_nms_ranges[j]
                lf_nms_threshold = lf_nms_thresholds[k]

                out = copy.copy(out_original)

                out = e2e_postprocessing.postprocess(out,
                    sol_threshold=sol_threshold,
                    lf_nms_params={
                        "overlap_range": lf_nms_range,
                        "overlap_threshold": lf_nms_threshold
                })
                order = e2e_postprocessing.read_order(out)
                e2e_postprocessing.filter_on_pick(out, order)

                e2e_postprocessing.trim_ends(out)

                preds = [decoded_hw[i] for i in out['idx']]
                pred = "\n".join(preds)

                error = error_rates.cer(gt, pred)

                results[key].append(error)

    sum_results = None
    if dataset_lookup == "validation_set":
        # Skipping because we didn't do the hyperparameter search
        sum_results = {}
        for k, v in results.iteritems():
            sum_results[k] = np.mean(v)

        sum_results = sorted(sum_results.iteritems(), key=operator.itemgetter(1))
        sum_results = sum_results[0]

    return sum_results, np.mean(aligned_results), np.mean(best_ever_results), sol, lf, hw

def main():
    config_path = sys.argv[1]

    with open(config_path) as f:
        config = yaml.load(f)

    mode = "all"
    if len(sys.argv) > 2:
        mode = sys.argv[2]


    start_idx = 0
    end_idx = 10
    if len(sys.argv) > 4:
        start_idx = int(sys.argv[3])
        end_idx = int(sys.argv[4])

    best_validation_so_far = None
    if mode in ['all', 'validation', 'init']:
        print "Running validation with best overall weight for baseline"
        error, i_error, mi_error, _, _, _ = alignment_step(config, dataset_lookup='validation_set', model_mode="best_overall")
        best_validation_so_far = error[1]
        print "Baseline Validation", error

    real_json_folder = config['training']['training_set']['json_folder']
    while True:

        for i in xrange(start_idx, end_idx):
            i_start = float(i) / config['training']['alignment']['train_refresh_groups']
            i_stop = float(i+1) / config['training']['alignment']['train_refresh_groups']

            if mode in ['all', 'training', 'init']:
                print ""
                print "Train running ", i
                start = time.time()
                error, i_error, mi_error, sol, lf, hw  = alignment_step(config, dataset_lookup='training_set', percent_range=[i_start, i_stop])
                print "Error:", error
                print "Ideal Error:", i_error
                print "Most Ideal Error:", mi_error
                print "Time:", time.time() - start

            if mode == 'init':
                #End early
                return

            if mode in ['all', 'validation']:
                print ""
                print "Test running"
                start = time.time()
                error, i_error, mi_error, sol, lf, hw = alignment_step(config, dataset_lookup='validation_set')
                if error[1] <= best_validation_so_far:
                    print "Saving best..."
                    dirname = config['training']['snapshot']['best_overall']
                    if not len(dirname) != 0 and os.path.exists(dirname):
                        os.makedirs(dirname)
                    save_path = os.path.join(dirname, "sol.pt")
                    torch.save(sol.state_dict(), save_path)
                    save_path = os.path.join(dirname, "lf.pt")
                    torch.save(lf.state_dict(), save_path)
                    save_path = os.path.join(dirname, "hw.pt")
                    torch.save(hw.state_dict(), save_path)
                    best_validation_so_far = error[1]

                print "Error:", error
                print "Ideal Error:", i_error
                print "Most Ideal Error:", mi_error
                print "Time:", time.time() - start


if __name__ == "__main__":
    main()
