import torch
import torch.nn as nn
from torch.autograd import Variable
from stn.gridgen import AffineGridGen, PerspectiveGridGen, GridGen
import numpy as np
from utils import transformation_utils
from lf_cnn import makeCnn
from fast_patch_view import get_patches

class LineFollower(nn.Module):
    def __init__(self, output_grid_size=32, dtype=torch.cuda.FloatTensor):
        super(LineFollower, self).__init__()
        cnn = makeCnn()
        position_linear = nn.Linear(512,5)
        position_linear.weight.data.zero_()
        position_linear.bias.data[0] = 0
        position_linear.bias.data[1] = 0
        position_linear.bias.data[2] = 0

        self.output_grid_size = output_grid_size

        self.dtype = dtype
        self.cnn = cnn
        self.position_linear = position_linear

    def forward(self, image, positions, steps=None, all_positions=[], reset_interval=-1, randomize=False, negate_lw=False, skip_grid=False, allow_end_early=False):

        batch_size = image.size(0)
        renorm_matrix = transformation_utils.compute_renorm_matrix(image)
        expanded_renorm_matrix = renorm_matrix.expand(batch_size,3,3)

        t = ((np.arange(self.output_grid_size) + 0.5) / float(self.output_grid_size))[:,None].astype(np.float32)
        t = np.repeat(t,axis=1, repeats=self.output_grid_size)
        t = Variable(torch.from_numpy(t), requires_grad=False).cuda()
        s = t.t()

        t = t[:,:,None]
        s = s[:,:,None]

        interpolations = torch.cat([
            (1-t)*s,
            (1-t)*(1-s),
            t*s,
            t*(1-s),
        ], dim=-1)

        view_window = Variable(torch.cuda.FloatTensor([
            [2,0,2],
            [0,2,0],
            [0,0,1]
        ])).expand(batch_size,3,3)

        step_bias = Variable(torch.cuda.FloatTensor([
            [1,0,2],
            [0,1,0],
            [0,0,1]
        ])).expand(batch_size,3,3)

        invert = Variable(torch.cuda.FloatTensor([
            [-1,0,0],
            [0,-1,0],
            [0,0,1]
        ])).expand(batch_size,3,3)

        if negate_lw:
            view_window = invert.bmm(view_window)

        grid_gen = GridGen(32,32)

        view_window_imgs = []
        next_windows = []
        reset_windows = True
        for i in xrange(steps):

            if i%reset_interval != 0 or reset_interval==-1:
                p_0 = positions[-1]

                if i == 0 and len(p_0.size()) == 3 and p_0.size()[1] == 3 and p_0.size()[2] == 3:
                    current_window = p_0
                    reset_windows = False
                    next_windows.append(p_0)

            else:
                p_0 = all_positions[i].type(self.dtype)
                reset_windows = True
                if randomize:
                    add_noise = p_0.clone()
                    add_noise.data.zero_()
                    mul_moise = p_0.clone()
                    mul_moise.data.fill_(1.0)

                    add_noise[:,0].data.uniform_(-2, 2)
                    add_noise[:,1].data.uniform_(-2, 2)
                    add_noise[:,2].data.uniform_(-.1, .1)

                    p_0 = p_0 * mul_moise + add_noise

            if reset_windows:
                reset_windows = False

                current_window = transformation_utils.get_init_matrix(p_0)

                if len(next_windows) == 0:
                    next_windows.append(current_window)
            else:
                current_window = next_windows[-1].detach()

            crop_window = current_window.bmm(view_window)

            resampled = get_patches(image, crop_window, grid_gen, allow_end_early)

            if resampled is None and i > 0:
                #get patches checks to see if stopping early is allowed
                break

            if resampled is None and i == 0:
                #Odd case where it start completely off of the edge
                #This happens rarely, but maybe should be more eligantly handled
                #in the future
                resampled = Variable(torch.zeros(crop_window.size(0), 3, 32, 32).type_as(image.data), requires_grad=False)


            # Process Window CNN
            cnn_out = self.cnn(resampled)
            cnn_out = torch.squeeze(cnn_out, dim=2)
            cnn_out = torch.squeeze(cnn_out, dim=2)
            delta = self.position_linear(cnn_out)


            next_window = transformation_utils.get_step_matrix(delta)
            next_window = next_window.bmm(step_bias)
            if negate_lw:
                next_window = invert.bmm(next_window).bmm(invert)

            next_windows.append(current_window.bmm(next_window))

        grid_line = []
        mask_line = []
        line_done = []
        xy_positions = []

        a_pt = Variable(torch.Tensor(
            [
                [0, 1,1],
                [0,-1,1]
            ]
        )).cuda()
        a_pt = a_pt.transpose(1,0)
        a_pt = a_pt.expand(batch_size, a_pt.size(0), a_pt.size(1))

        for i in xrange(0, len(next_windows)-1):

            w_0 = next_windows[i]
            w_1 = next_windows[i+1]

            pts_0 = w_0.bmm(a_pt)
            pts_1 = w_1.bmm(a_pt)
            xy_positions.append(pts_0)

            if skip_grid:
                continue

            pts = torch.cat([pts_0, pts_1], dim=2)

            grid_pts = expanded_renorm_matrix.bmm(pts)

            grid = interpolations[None,:,:,None,:] * grid_pts[:,None,None,:,:]
            grid = grid.sum(dim=-1)[...,:2]

            grid_line.append(grid)

        xy_positions.append(pts_1)

        if skip_grid:
            grid_line = None
        else:
            grid_line = torch.cat(grid_line, dim=1)

        return grid_line, view_window_imgs, next_windows, xy_positions
