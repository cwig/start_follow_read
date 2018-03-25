import torch
from torch.autograd import Variable

from utils import transformation_utils

def get_patches(image, crop_window, grid_gen, allow_end_early=False):


        pts = Variable(torch.FloatTensor([
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [ 1.0, 1.0,  1.0, 1.0]
        ]).type_as(image.data), requires_grad=False)[None,...]

        bounds = crop_window.matmul(pts)

        min_bounds, _ = bounds.min(dim=-1)
        max_bounds, _ = bounds.max(dim=-1)
        d_bounds = max_bounds - min_bounds
        floored_idx_offsets = torch.floor(min_bounds[:,:2].data).long()
        max_d_bounds = d_bounds.max(dim=0)[0].max(dim=0)[0]
        crop_size = torch.ceil(max_d_bounds).long()
        if image.is_cuda:
            crop_size = crop_size.cuda()
        w = crop_size.data[0]

        memory_space = Variable(torch.zeros(d_bounds.size(0), 3, w, w).type_as(image.data), requires_grad=False)
        translations = []
        N = transformation_utils.compute_renorm_matrix(memory_space)
        all_skipped = True

        for b_i in xrange(memory_space.size(0)):

            o = floored_idx_offsets[b_i]

            t = Variable(torch.cuda.FloatTensor([
                [1,0,-o[0]],
                [0,1,-o[1]],
                [0,0,    1]
            ]), requires_grad=False).expand(3,3)
            translations.append(N.mm(t)[None,...])

            skip_slice = False

            s_x = (o[0], o[0]+w)
            s_y = (o[1], o[1]+w)
            t_x = (0, w)
            t_y = (0, w)
            if o[0] < 0:
                s_x = (0, w+o[0])
                t_x = (-o[0], w)

            if o[1] < 0:
                s_y = (0, w+o[1])
                t_y = (-o[1], w)

            if o[0]+w >= image.size(2):
                s_x = (s_x[0], image.size(2))
                t_x = (t_x[0], image.size(2) - s_x[0])

            if o[1]+w >= image.size(3):
                s_y = (s_y[1], image.size(3))
                t_y = (t_y[1], image.size(3) - s_y[1])

            if s_x[0] >= s_x[1]:
                skip_slice = True

            if t_x[0] >= t_x[1]:
                skip_slice = True

            if s_y[0] >= s_y[1]:
                skip_slice = True

            if t_y[0] >= t_y[1]:
                skip_slice = True

            if not skip_slice:
                all_skipped = False
                i_s  = image[b_i:b_i+1, :, s_x[0]:s_x[1], s_y[0]:s_y[1]]
                memory_space[b_i:b_i+1, :, t_x[0]:t_x[1], t_y[0]:t_y[1]] = i_s

        if all_skipped and allow_end_early:
            return None

        translations = torch.cat(translations, 0)
        grid = grid_gen(translations.bmm(crop_window))
        grid = grid[:,:,:,0:2] / grid[:,:,:,2:3]

        resampled = torch.nn.functional.grid_sample(memory_space.transpose(2,3), grid, mode='bilinear')

        return resampled
