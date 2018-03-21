from utils import string_utils, error_rates
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
from torch.autograd import Variable
# from warpctc_pytorch import CTCLoss
#
# criterion = CTCLoss()

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


def alignment(predictions, hw_scores):

    # confidences = predictions[:,:,4]
    #
    # log_confidences = torch.log(confidences + 1e-10)
    # log_one_minus_confidences = torch.log(1.0 - confidences + 1e-10)

    # expanded_log_confidences = log_confidences[:,:,None].expand(confidences.size(0), confidences.size(1), hw_scores.size(2))
    # expanded_log_one_minus_confidences = log_one_minus_confidences[:,:,None].expand(confidences.size(0), confidences.size(1), hw_scores.size(2))

    # C = alpha_alignment * hw_scores - expanded_log_confidences + expanded_log_one_minus_confidences
    C = hw_scores

    C = C.data.cpu().numpy()
    X = np.zeros_like(C)

    idxs = []
    for b in xrange(C.shape[0]):
        C_i = C[b]
        row_ind, col_ind = linear_sum_assignment(C_i.T)
        idxs.append((col_ind, row_ind))
        X[b][(col_ind, row_ind)] = 1.0

    X = Variable(torch.from_numpy(X).type(predictions.data.type()), requires_grad=False)
    X2 = 1.0 - torch.sum(X, 2)

    init_hw_loss = (hw_scores * X).sum() / X.sum()

    # selected_confidences = (expanded_log_confidences * X).sum() / hw_scores.size(2)
    # confidence_loss =  -100*(expanded_log_confidences * X).sum() - (log_one_minus_confidences * X2).sum()

    return idxs, init_hw_loss

def loss(predictions, aligned_idxs):
    pass
