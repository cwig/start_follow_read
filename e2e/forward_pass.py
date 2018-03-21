from e2e import e2e_model
from e2e.e2e_model import E2EModel

import validation_utils

from utils import error_rates

import itertools
import copy
import numpy as np
import cv2

def forward_pass(x, e2e, config, thresholds, idx_to_char, update_json=False):

    gt_lines = x['gt_lines']
    gt = "\n".join(gt_lines)

    out_original = e2e(x)
    results = {}
    if out_original is None:
        #TODO: not a good way to handle this, but fine for now
        None

    gt_lines = x['gt_lines']
    gt = "\n".join(gt_lines)

    out_original = E2EModel.results_to_numpy(out_original)
    out_original['idx'] = np.arange(out_original['sol'].shape[0])

    decoded_hw, decoded_raw_hw = E2EModel.decode_handwriting(out_original, idx_to_char)
    pick, costs = E2EModel.align_to_gt_lines(decoded_hw, gt_lines)

    most_ideal_pred_lines, improved_idxs = validation_utils.update_ideal_results(pick, costs, decoded_hw, x['gt_json'])
    # if update_json:
    #     validation_utils.save_improved_idxs(improved_idxs, decoded_hw,
    #                                         decoded_raw_hw, out_original,
    #                                         x, config[dataset_lookup]['json_folder'], config['alignment']['trim_to_sol'])

    sol_thresholds = thresholds[0]
    sol_thresholds_idx = range(len(sol_thresholds))

    lf_nms_ranges =  thresholds[1]
    lf_nms_ranges_idx = range(len(lf_nms_ranges))

    lf_nms_thresholds = thresholds[2]
    lf_nms_thresholds_idx = range(len(lf_nms_thresholds))

    most_ideal_pred_lines = "\n".join(most_ideal_pred_lines)

    ideal_pred_lines = [decoded_hw[i] for i in pick]
    ideal_pred_lines = "\n".join(ideal_pred_lines)

    error = error_rates.cer(gt, ideal_pred_lines)
    ideal_result = error

    error = error_rates.cer(gt, most_ideal_pred_lines)
    most_ideal_result = error

    for key in itertools.product(sol_thresholds_idx, lf_nms_ranges_idx, lf_nms_thresholds_idx):
        i,j,k = key
        sol_threshold = sol_thresholds[i]
        lf_nms_range = lf_nms_ranges[j]
        lf_nms_threshold = lf_nms_thresholds[k]

        out = copy.copy(out_original)

        out = E2EModel.postprocess(out,
            sol_threshold=sol_threshold,
            lf_nms_params={
                "overlap_range": lf_nms_range,
                "overlap_threshold": lf_nms_threshold
        })
        order = E2EModel.read_order(out)
        E2EModel.filter_on_pick(out, order)

        # draw_img = E2EModel.draw_output(out, img)
        # cv2.imwrite("test_b_samples/test_img_{}.png".format(a), draw_img)

        preds = [decoded_hw[i] for i in out['idx']]
        pred = "\n".join(preds)

        error = error_rates.cer(gt, pred)

        results[key] = error

    return results, ideal_result, most_ideal_result
