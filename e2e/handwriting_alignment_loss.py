from utils import string_utils, error_rates
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
from torch.autograd import Variable


def accumulate_scores(out, out_positions, xy_positions, gt_state, idx_to_char):


    preds = out.transpose(0,1).cpu()
    batch_size = preds.size(1)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

    for i, logits in enumerate(out.data.cpu().numpy()):
        raw_decode, raw_decode_full = string_utils.naive_decode(logits)
        pred_str = string_utils.label2str_single(raw_decode, idx_to_char, False)
        pred_str_full = string_utils.label2str_single(raw_decode_full, idx_to_char, True)


        sub_out_positions = [o[i].data.cpu().numpy().tolist() for o in out_positions]
        sub_xy_positions = [o[i].data.cpu().numpy().tolist() for o in xy_positions]

        for gt_obj in gt_state:
            gt_text = gt_obj['gt']
            cer = error_rates.cer(gt_text, pred_str)

            #This is a terrible way to do this...
            gt_obj['errors'] = gt_obj.get('errors', [])
            gt_obj['pred'] = gt_obj.get('pred', [])
            gt_obj['pred_full'] = gt_obj.get('pred_full', [])
            gt_obj['path'] = gt_obj.get('path', [])
            gt_obj['path_xy'] = gt_obj.get('path_xy', [])

            gt_obj['errors'].append(cer)
            gt_obj['pred'].append(pred_str)
            gt_obj['pred_full'].append(pred_str_full)
            gt_obj['path'].append(sub_out_positions)
            gt_obj['path_xy'].append(sub_xy_positions)


def update_alignment(out, gt_lines, alignments, idx_to_char, idx_mapping, sol_positions):

    preds = out.cpu()
    batch_size = preds.size(1)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

    for i, logits in enumerate(out.data.cpu().numpy()):
        raw_decode, raw_decode_full = string_utils.naive_decode(logits)
        pred_str = string_utils.label2str_single(raw_decode, idx_to_char, False)

        for j, gt in enumerate(gt_lines):
            cer = error_rates.cer(gt, pred_str)
            global_i = idx_mapping[i]
            c = sol_positions[i,0,-1].data[0]

            # alignment_error = cer
            alignment_error = cer + 0.1 * (1.0 - c)

            if alignment_error < alignments[j][0]:
                alignments[j][0] = alignment_error
                alignments[j][1] = global_i
                # alignments[j][2] = out[i][:,None,:]
                alignments[j][2] = None
                alignments[j][3] = pred_str

def alignment(predictions, hw_scores, alpha_alignment=0.1, alpha_backprop=0.1):
    confidences = predictions[:,:,4]

    log_confidences = torch.log(confidences + 1e-10)
    log_one_minus_confidences = torch.log(1.0 - confidences + 1e-10)

    expanded_log_confidences = log_confidences[:,:,None].expand(confidences.size(0), confidences.size(1), hw_scores.size(2))
    expanded_log_one_minus_confidences = log_one_minus_confidences[:,:,None].expand(confidences.size(0), confidences.size(1), hw_scores.size(2))

    C = alpha_alignment * hw_scores - expanded_log_confidences + expanded_log_one_minus_confidences

    C = C.data.cpu().numpy()
    X = np.zeros_like(C)

    idxs = []
    for b in xrange(C.shape[0]):
        C_i = C[b]
        row_ind, col_ind = linear_sum_assignment(C_i.T)
        idxs.append((col_ind, row_ind))

    return idxs

def loss(preds, non_hw_sol, hw_sol, gt_lines, char_to_idx, criterion):
    label_lengths = []
    all_labels = []
    for gt_str in gt_lines:
        l = string_utils.str2label_single(gt_str, char_to_idx)
        all_labels.append(l)
        label_lengths.append(len(l))

    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    labels = torch.from_numpy(all_labels.astype(np.int32))
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    labels =  Variable(labels, requires_grad=False)
    label_lengths = Variable(label_lengths, requires_grad=False)

    batch_size = preds.size(0)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

    ctc_loss = 1e-2 * criterion(preds.cpu(), labels, preds_size, label_lengths)

    log_one_minus_confidences = torch.log(1.0 - non_hw_sol[:,:,0] + 1e-10)
    log_confidences = torch.log(hw_sol[:,:,0] + 1e-10)

    selected_confidence = log_confidences.sum()
    not_selected_confidence = log_one_minus_confidences.sum()

    confidence_loss = -selected_confidence - not_selected_confidence

    # print " - - - - Losses - - - - "
    # print ctc_loss.data[0]
    # print selected_confidence.data[0], log_confidences.size()
    # print not_selected_confidence.data[0], log_one_minus_confidences.size()
    # print ""

    return ctc_loss + confidence_loss.cpu()
