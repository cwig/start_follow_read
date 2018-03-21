import numpy as np
import torch
from torch.autograd import Variable
import handwriting_alignment_loss

from lf import transformation_utils

def run_forward_pass(x, sol, lf, hw, idx_to_char, use_full_img=False, accpet_threshold=0.1, dtype=None):
    gt_state = [{
        'gt': gt
    } for gt in x['gt_lines']]

    sol_img = Variable(x['resized_img'].type(dtype), requires_grad=False, volatile=True)

    if use_full_img:
        img = Variable(x['full_img'].type(dtype), requires_grad=False, volatile=True)
        scale = x['resize_scale']
        results_scale = 1.0
    else:
        img = sol_img
        scale = 1.0
        results_scale = x['resize_scale']

    start = sol(sol_img)

    sorted_start, sorted_indices = torch.sort(start[...,0:1], dim=1, descending=True)
    min_threshold = sorted_start[0,len(gt_state),0].data.cpu()[0]

    threshold = min(accpet_threshold, min_threshold)
    select = start[...,0:1] > threshold

    select = select.expand(select.size(0), select.size(1), start.size(2))
    start = start[select].view(start.size(0), -1, start.size(2))

    perform_forward = len(start.size()) == 3
    if not perform_forward:
        return float(len(gt_state))

    start = start.transpose(0,1)

    rot_val = torch.zeros(start.size(0),1,1).type(dtype)
    rot_val = Variable(rot_val, requires_grad=False, volatile=True)

    # print start[...,4:5]

    positions = torch.cat([
        start[...,1:3]  * scale,
        start[...,3:4],
        start[...,4:5]  * scale,
        start[...,0:1]
    ], 2)

    new_start_positions = None
    p_interval = 128
    for p in xrange(0,min(positions.size(0), np.inf), p_interval):
        sub_positions = positions[p:p+p_interval,0,:]
        batch_size = sub_positions.size(0)
        sub_positions = [sub_positions]

        expand_img = img.expand(sub_positions[0].size(0), img.size(1), img.size(2), img.size(3))

        step_size = 5
        extra_bw = 1
        grid_line, _, out_positions, _ = lf(expand_img, sub_positions, steps=step_size)
        grid_line, _, out_positions, _ = lf(expand_img, [out_positions[step_size]], steps=step_size+extra_bw, negate_lw=True)
        grid_line, _, out_positions, xy_positions = lf(expand_img, [out_positions[step_size+extra_bw]], steps=20)


        if new_start_positions is None:
            new_start_positions = out_positions[extra_bw]
        else:
            new_start_positions = torch.cat([
                new_start_positions,
                out_positions[extra_bw]
            ],0)


        expand_img = expand_img.transpose(2,3)

        hw_interval = 128
        for h in xrange(0,min(grid_line.size(0), np.inf), hw_interval):

            sub_out_positions = [o[h:h+hw_interval] for o in out_positions]
            sub_xy_positions = [o[h:h+hw_interval] for o in xy_positions]

            line = torch.nn.functional.grid_sample(expand_img[h:h+hw_interval], grid_line[h:h+hw_interval])
            line = line.transpose(2,3)

            #for l in line:
            #    l = l.transpose(0,1).transpose(1,2)
            #    l = (l + 1)*128
            #    l_np = l.data.cpu().numpy()
            #    cv2.imwrite("debug/{}.png".format(cnt), l_np)
            #    cnt+= 1

            #print "Here..."
            #raw_input()

            out = hw(line)
            out = out.transpose(0,1)

            handwriting_alignment_loss.accumulate_scores(out, sub_out_positions, sub_xy_positions, gt_state, idx_to_char)

    hw_scores = np.array([d['errors'] for d in gt_state]).astype(np.float32).T[None,...]
    hw_scores = Variable(torch.from_numpy(hw_scores), requires_grad=False).cuda()

    new_start_positions = new_start_positions[:,None,:]
    aligned_idxs, hw_loss = handwriting_alignment_loss.alignment(new_start_positions, hw_scores)

    return float(hw_loss.data[0])
