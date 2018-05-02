from utils import string_utils, error_rates
import numpy as np
import nms
import copy

def get_trimmed_polygons(out):
    all_polygons = []
    for j in xrange(out['lf'][0].shape[0]):
        begin = out['beginning'][j]
        end = out['ending'][j]
        last_xy = None
        begin_f = int(np.floor(begin))
        end_f = int(np.ceil(end))
        points = []
        for i in xrange(begin_f, end_f+1):

            if i == begin_f:
                p0 = out['lf'][i][j]
                p1 = out['lf'][i+1][j]
                t = begin - np.floor(begin)
                p = p0 * (1 - t) + p1 * t

            elif i == end_f:

                p0 = out['lf'][i-1][j]
                if i != len(out['lf']):
                    p1 = out['lf'][i][j]
                    t = end - np.floor(end)
                    p = p0 * (1 - t) + p1 * t
                else:
                    p = p0
            else:
                p = out['lf'][i][j]

            points.append(p)
        points = np.array(points)
        all_polygons.append(points)
    return all_polygons

def trim_ends(out):

    lf_length = len(out['lf'])
    hw = out['hw']
    selected = hw.argmax(axis=-1)
    beginning = np.argmax(selected != 0, axis=1)
    ending = selected.shape[1] - 1 - np.argmax(selected[:,::-1] != 0, axis=1)

    beginning_percent = (beginning+0.5) / float(selected.shape[1])
    ending_percent = (ending+0.5) / float(selected.shape[1])

    lf_beginning = lf_length * beginning_percent
    lf_ending = lf_length * ending_percent

    out['beginning'] = lf_beginning
    out['ending'] = lf_ending
    return out

def filter_on_pick(out, pick):
    out['sol'] = out['sol'][pick]
    out['lf'] = [l[pick] for l in out['lf']]
    out['hw'] = out['hw'][pick]
    if 'idx' in out:
        out['idx'] = out['idx'][pick]
    if 'beginning' in out:
        out['beginning'] = out['beginning'][pick]
    if 'ending' in out:
        out['ending'] = out['ending'][pick]

def filter_on_pick_no_copy(out, pick):
    output = {}
    output['sol'] = out['sol'][pick]
    output['lf'] = [l[pick] for l in out['lf']]
    output['hw'] = out['hw'][pick]
    if 'idx' in out:
        output['idx'] = out['idx'][pick]
    if 'beginning' in out:
        output['beginning'] = out['beginning'][pick]
    if 'ending' in out:
        output['ending'] = out['ending'][pick]
    return output

def select_non_empty_string(out):
    selected = out['hw'].argmax(axis=-1)
    return np.where(selected.sum(axis=1) != 0)

def postprocess(out, **kwargs):
    out = copy.copy(out)

    # postprocessing should be done with numpy data
    sol_threshold = kwargs.get("sol_threshold", None)
    sol_nms_threshold = kwargs.get("sol_nms_threshold", None)
    lf_nms_params = kwargs.get('lf_nms_params', None)
    lf_nms_2_params = kwargs.get('lf_nms_2_params', None)

    if sol_threshold is not None:
        pick = np.where(out['sol'][:,-1] > sol_threshold)
        filter_on_pick(out, pick)

    if sol_nms_threshold is not None:
        raise Exception("This is not correct")
        pick = nms.sol_nms_single(out['sol'], sol_nms_threshold)
        out['sol'] =  out['sol'][pick]

    if lf_nms_params is not None:
        confidences = out['sol'][:,-1]
        overlap_range = lf_nms_params['overlap_range']
        overlap_thresh = lf_nms_params['overlap_threshold']

        lf_setup = np.concatenate([l[None,...] for l in out['lf']])
        lf_setup = [lf_setup[:,i] for i in range(lf_setup.shape[1])]

        pick = nms.lf_non_max_suppression_area(lf_setup, confidences, overlap_range, overlap_thresh)
        filter_on_pick(out, pick)

    if lf_nms_2_params is not None:
        confidences = out['sol'][:,-1]
        overlap_thresh = lf_nms_2_params['overlap_threshold']
        refined_lf = get_trimmed_polygons(out)
        pick = nms.lf_non_max_suppression_area(refined_lf, confidences, None, overlap_thresh)
        filter_on_pick(out, pick)

    return out

def read_order(out):
    first_pt = out['lf'][0][:,:2,0]

    first_pt = first_pt[:,::-1]
    first_pt = np.concatenate([first_pt, np.arange(first_pt.shape[0])[:,None]], axis=1)
    first_pt = first_pt.tolist()

    first_pt.sort()

    return [int(p[2]) for p in first_pt]

def decode_handwriting(out, idx_to_char):
    hw_out = out['hw']
    list_of_pred = []
    list_of_raw_pred = []
    for i in xrange(hw_out.shape[0]):
        logits = hw_out[i,...]
        pred, raw_pred = string_utils.naive_decode(logits)
        pred_str = string_utils.label2str_single(pred, idx_to_char, False)
        raw_pred_str = string_utils.label2str_single(raw_pred, idx_to_char, True)
        list_of_pred.append(pred_str)
        list_of_raw_pred.append(raw_pred_str)

    return list_of_pred, list_of_raw_pred


def results_to_numpy(out):
    return {
        "sol": out['sol'].data.cpu().numpy()[:,0,:],
        "lf": [l.data.cpu().numpy() for l in out['lf']] if out['lf'] is not None else None,
        "hw": out['hw'].data.cpu().numpy(),
        "results_scale": out['results_scale'],
        "line_imgs": out['line_imgs'],
    }

def align_to_gt_lines(decoded_hw, gt_lines):
    costs = []
    for i in xrange(len(decoded_hw)):
        costs.append([])
        for j in xrange(len(gt_lines)):
            pred = decoded_hw[i]
            gt = gt_lines[j]
            cer = error_rates.cer(gt, pred)
            costs[i].append(cer)

    costs = np.array(costs)
    min_idx = costs.argmin(axis=0)
    min_val = costs.min(axis=0)

    return min_idx, min_val
