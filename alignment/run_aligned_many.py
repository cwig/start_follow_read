import torch
from torch.autograd import Variable
import classic_mod
from sol import start_of_line_finder
from lf import line_follower
import sys
import os
import cv2
import numpy as np
import string_utils
import json
import error_rates
import time
import handwriting_alignment_loss

with open(sys.argv[2]) as f:
    char_set = json.load(f)


idx_to_char = {}
for k,v in char_set['idx_to_char'].iteritems():
    idx_to_char[int(k)] = v

dtype = torch.cuda.FloatTensor

#sol_state = torch.load('snapshots_1_20/sol3.pt')
#sol_state = torch.load('snapshots_1_23/sol3.pt')
sol_state = torch.load('new_snapshots/sol.pt')
line_state = torch.load('snapshots_1_26/lf.pt')
#line_state = torch.load('new_snapshots/lf.pt')
# line_state = torch.load('snapshots_1_20/lf_2.pt')
hw_state = torch.load('e2e_snapshots/init/hw-snapshot.pt')

base_0 = 1/16.0
base_1 = 1/16.0
base_scale = 16.0

sol = start_of_line_finder.StartOfLineFinder(base_0, base_1, base_scale)
sol.load_state_dict(sol_state)

#lf = simple_line_follower.SimpleLineFollower()
lf = line_follower.LineFollower()
lf.load_state_dict(line_state)

hw = classic_mod.create_model({
    "inputHeight": 32,
    "numOfChannels": 3,
    "numOfOutputs": 197
})
hw.load_state_dict(hw_state)

sol.eval()
lf.eval()
hw.eval()

with open(sys.argv[3]) as f:
    gt_labels = json.load(f)


for img_num in range(10001,15001):
    print "ID",img_num

    if os.path.isfile(os.path.join(sys.argv[5], "0{}.json".format(img_num))):
        continue

    img_path = "../hwn5-comp-2017/data/Train-B/batch1/0{}.jpg".format(img_num)
    img_file_name = img_path.split("/")[-1]
    gt_lines = None
    for v in gt_labels:
        gt_file_path = v['image_path'].split("/")[-1]
        if gt_file_path == img_file_name:
            gt_lines = v['gt_lines']
            break

    if gt_lines is None:
        print "Unable to file gt"
        continue

    gt_results = [
    {
        'gt':v,
        'bsf':np.inf,
        'errors':[],
        'pred':[],
        'pred_full':[],
        'path':[]
    } for v in gt_lines]

    org_img = cv2.imread(img_path)

    target_dim1 = 512
    s = target_dim1 / float(org_img.shape[1])
    target_dim0 = int(org_img.shape[0]/float(org_img.shape[1]) * target_dim1)
    org_img = cv2.resize(org_img,(target_dim1, target_dim0), interpolation = cv2.INTER_CUBIC)
    img = org_img
    img = img.transpose([2,1,0])[None,...]

    img = Variable(torch.from_numpy(img).type(dtype), requires_grad=False, volatile=True)
    img = img / 128 - 1

    start = sol(img)



    select = start[...,0:1] > 0.1
    start = torch.cat([
        start[...,0:1][select][None,:,None],
        start[...,1:2][select][None,:,None],
        start[...,2:3][select][None,:,None],
        start[...,3:4][select][None,:,None],
        start[...,4:5][select][None,:,None]
    ], 2)

    start = start.transpose(0,1)
    print "Start Size:", start.size()

    rot_val = torch.zeros(start.size(0),1,1).type(dtype)
    rot_val = Variable(rot_val, requires_grad=False, volatile=True)

    positions = torch.cat([
       start[...,1:3],
       rot_val,
       # start[...,4:5],
       torch.abs(start[...,3:4]),
       start[...,0:1]
    ], 2)


    # positions = torch.cat([
    #     start[...,2:4],
    #     start[...,5:6],
    #     start[...,4:5],
    #     start[...,0:1]
    # ], 2)


    all_lines = np.zeros_like(org_img)
    p_interval = 512
    # p_interval = positions.size(0)
    start = time.time()

    new_start_positions = None

    cnt = 0
    for p in xrange(0,min(positions.size(0), np.inf), p_interval):
        print "P", p, p_interval
        sub_positions = positions[p:p+p_interval,0,:]
        batch_size = sub_positions.size(0)
        sub_positions = [sub_positions]

        expand_img = img.expand(sub_positions[0].size(0), img.size(1), img.size(2), img.size(3))

        step_size = 5
        extra_bw = 1
        grid_line, _, _, out_positions = lf(expand_img, sub_positions, steps=step_size)
        grid_line, _, _, out_positions = lf(expand_img, [out_positions[step_size]], steps=step_size+extra_bw, negate_lw=True)
        grid_line, _, _, out_positions = lf(expand_img, [out_positions[step_size+extra_bw]], steps=20)

        print "Out Position Size:", out_positions[0].size()
        if new_start_positions is None:
            new_start_positions = out_positions[extra_bw]
        else:
            new_start_positions = torch.cat([
                new_start_positions,
                out_positions[extra_bw]
            ],0)


        expand_img = expand_img.transpose(2,3)
        print "TIME1", time.time() - start

        hw_interval = 256
        for h in xrange(0,min(grid_line.size(0), np.inf), hw_interval):

            sub_out_positions = [o[h:h+hw_interval] for o in out_positions]

            line = torch.nn.functional.grid_sample(expand_img[h:h+hw_interval], grid_line[h:h+hw_interval])
            line = line.transpose(2,3)

            for l in line:
                l = l.transpose(0,1).transpose(1,2)
                l = (l + 1)*128
                l_np = l.data.cpu().numpy()
                cv2.imwrite("debug/{}.png".format(cnt), l_np)
                cnt+= 1


            out = hw(line)
            out = out.transpose(0,1)

            handwriting_alignment_loss.accumulate_scores(out, sub_out_positions, gt_results, idx_to_char)


    hw_scores = np.array([d['errors'] for d in gt_results]).astype(np.float32).T[None,...]
    hw_scores = Variable(torch.from_numpy(hw_scores), requires_grad=False).cuda()


    new_start_positions = new_start_positions[:,None,:]
    aligned_idxs, confidence_loss, init_hw_loss, selected_confidences = handwriting_alignment_loss.alignment(new_start_positions, hw_scores, 1.0)
    print init_hw_loss.data.cpu().numpy()

    start = new_start_positions.data.cpu().numpy()
    valid_starts = start[(aligned_idxs[0][0]),...]

    output_lines = []
    good_lines = np.zeros_like(org_img)
    a = []
    for i, j in zip(*aligned_idxs[0]):
        zero_img = np.zeros_like(org_img)
        color = (1,1,1)

        gt_obj = gt_results[j]
        aln_path = gt_obj['path'][i][extra_bw:]

        a.append( min(gt_obj['errors']) )

        for k in range(len(aln_path)-1):
            p1 = aln_path[k]
            p2 = aln_path[k+1]
            scale = int(p1[3])

            x1, y1 = p1[:2]
            x2, y2 = p2[:2]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.line(zero_img,(x1,y1),(x2,y2),color,1)

            if k == 0:
                scale = max(1,scale)
                cv2.circle(zero_img, (x1,y1), scale, color, 1)

        good_lines = np.maximum(good_lines, zero_img)

        output_lines.append({
            'gt': gt_obj['gt'],
            'pred': gt_obj['pred'][i],
            'pred_full': gt_obj['pred_full'][i],
            'path': gt_obj['path'][i],
            'sol': gt_obj['path'][i][extra_bw],
            'start_idx': extra_bw,
            'scaler': s
        })


    org_img[good_lines != 0] = good_lines[good_lines != 0]
    cv2.imwrite(os.path.join(sys.argv[4], "1{}.png".format(img_num)), org_img)

    with open(os.path.join(sys.argv[5], "1{}.json".format(img_num)), 'w') as f:
        json.dump(output_lines, f)
