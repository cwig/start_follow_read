import torch
import torch.nn as nn
from torch.autograd import Variable

import cv2
import numpy as np
from utils import string_utils, error_rates
from utils import transformation_utils
import handwriting_alignment_loss

import e2e_postprocessing

import copy
from scipy.optimize import linear_sum_assignment
import math

class E2EModel(nn.Module):
    def __init__(self, sol, lf, hw, dtype=torch.cuda.FloatTensor):
        super(E2EModel, self).__init__()

        self.dtype = dtype

        self.sol = sol
        self.lf = lf
        self.hw = hw



    def train(self):
        self.sol.train()
        self.lf.train()
        self.hw.train()

    def eval(self):
        self.sol.eval()
        self.lf.eval()
        self.hw.eval()

    def forward(self, x, use_full_img=True, accpet_threshold=0.1, volatile=True, gt_lines=None, idx_to_char=None):

        sol_img = Variable(x['resized_img'].type(self.dtype), requires_grad=False, volatile=volatile)

        if use_full_img:
            img = Variable(x['full_img'].type(self.dtype), requires_grad=False, volatile=volatile)
            scale = x['resize_scale']
            results_scale = 1.0
        else:
            img = sol_img
            scale = 1.0
            results_scale = x['resize_scale']

        original_starts = self.sol(sol_img)

        start = original_starts

        #Take at least one point
        sorted_start, sorted_indices = torch.sort(start[...,0:1], dim=1, descending=True)
        min_threshold = sorted_start[0,1,0].data.cpu()[0]
        accpet_threshold = min(accpet_threshold, min_threshold)

        select = original_starts[...,0:1] >= accpet_threshold

        select_idx = np.where(select.data.cpu().numpy())[1]

        select = select.expand(select.size(0), select.size(1), start.size(2))
        select = select.detach()
        start = start[select].view(start.size(0), -1, start.size(2))

        perform_forward = len(start.size()) == 3

        if not perform_forward:
            return None

        forward_img = img

        start = start.transpose(0,1)

        positions = torch.cat([
           start[...,1:3]  * scale,
           start[...,3:4],
           start[...,4:5]  * scale,
           start[...,0:1]
        ], 2)

        hw_out = []
        p_interval = positions.size(0)
        lf_xy_positions = None
        line_imgs = []
        for p in xrange(0,min(positions.size(0), np.inf), p_interval):
            sub_positions = positions[p:p+p_interval,0,:]
            sub_select_idx = select_idx[p:p+p_interval]

            batch_size = sub_positions.size(0)
            sub_positions = [sub_positions]

            expand_img = forward_img.expand(sub_positions[0].size(0), img.size(1), img.size(2), img.size(3))

            step_size = 5
            extra_bw = 1
            forward_steps = 40

            grid_line, _, out_positions, xy_positions = self.lf(expand_img, sub_positions, steps=step_size)
            grid_line, _, out_positions, xy_positions = self.lf(expand_img, [out_positions[step_size]], steps=step_size+extra_bw, negate_lw=True)
            grid_line, _, out_positions, xy_positions = self.lf(expand_img, [out_positions[step_size+extra_bw]], steps=forward_steps, allow_end_early=True)

            if lf_xy_positions is None:
                lf_xy_positions = xy_positions
            else:
                for i in xrange(len(lf_xy_positions)):
                    lf_xy_positions[i] = torch.cat([
                        lf_xy_positions[i],
                        xy_positions[i]
                    ])
            expand_img = expand_img.transpose(2,3)

            hw_interval = p_interval
            for h in xrange(0,min(grid_line.size(0), np.inf), hw_interval):
                sub_out_positions = [o[h:h+hw_interval] for o in out_positions]
                sub_xy_positions = [o[h:h+hw_interval] for o in xy_positions]
                sub_sub_select_idx = sub_select_idx[h:h+hw_interval]

                line = torch.nn.functional.grid_sample(expand_img[h:h+hw_interval].detach(), grid_line[h:h+hw_interval])
                line = line.transpose(2,3)

                for l in line:
                    l = l.transpose(0,1).transpose(1,2)
                    l = (l + 1)*128
                    l_np = l.data.cpu().numpy()
                    line_imgs.append(l_np)
                #     cv2.imwrite("example_line_out.png", l_np)
                #     print "Saved!"
                #     raw_input()

                out = self.hw(line)
                out = out.transpose(0,1)

                hw_out.append(out)


        hw_out = torch.cat(hw_out, 0)


        return {
            "original_sol": original_starts,
            "sol": positions,
            "lf": lf_xy_positions,
            "hw": hw_out,
            "results_scale": results_scale,
            "line_imgs": line_imgs
        }
