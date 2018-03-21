import torch
import numpy as np
import pyclipper

def sol_non_max_suppression(start_torch, overlap_thresh):
    #Todo: Make this work with batches

    #Rotation is not taken into account
    start = start_torch.data.cpu().numpy()

    pick = sol_nms_single(start[0], overlap_thresh)

    zero_idx = [0 for _ in xrange(len(pick))]

    select = (zero_idx, pick)
    return start_torch[select][None,...]


def sol_nms_single(start, overlap_thresh):
    # Based on https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # Maybe could port to pytorch to work over the tensors directly

    x1 = start[:,1] - start[:,3]
    y1 = start[:,2] - start[:,3]

    x2 = start[:,1] + start[:,3]
    y2 = start[:,2] + start[:,3]

    c = start[:,0]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(c)

    pick = []
    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))
    return pick

def lf_non_max_suppression_area(lf_xy_positions, confidences, overlap_range, overlap_thresh):
    # lf_xy_positions = np.concatenate([l.data.cpu().numpy()[None,...] for l in lf_xy_positions])
    # lf_xy_positions = lf_xy_positions[:,:,:2,:2]

    # print lf_xy_positions
    # raw_input()
    lf_xy_positions = [l[:,:2,:2] for l in lf_xy_positions]
    #this assumes equal length positions
    # lf_xy_positions = np.concatenate([l[None,...] for l in lf_xy_positions])
    # lf_xy_positions = lf_xy_positions[:,:,:2,:2]

    c = confidences


    bboxes = []
    center_lines = []
    scales = []
    for i in xrange(len(lf_xy_positions)):
        pts = lf_xy_positions[i]
    # for i in xrange(lf_xy_positions.shape[1]):
        # pts = lf_xy_positions[:,i,:]
        if overlap_range is not None:
            pts = pts[overlap_range[0]: overlap_range[1]]

        f = pts[0]
        delta = f[:,0] - f[:,1]
        scale = np.sqrt( (delta**2).sum() )
        scales.append(scale)

        # ls = pts[:,:,0].tolist() + pts[:,:,1][::-1].tolist()
        # ls = [[int(x[0]), int(x[1])] for x in ls]
        # poly_regions.append(ls)
        center_lines.append( (pts[:,:,0] + pts[:,:,1])/2.0 )

        min_x = pts[:,0].min()
        max_x = pts[:,0].max()
        min_y = pts[:,1].min()
        max_y = pts[:,1].max()

        bboxes.append((min_x, min_y, max_x, max_y))

    bboxes = np.array(bboxes)

    if len(bboxes.shape) < 2:
        return []

    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(c)


    overlapping_regions = []
    pick = []
    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap_bb = (w * h) / area[idxs[:last]]

        overlap = []
        for step, j in enumerate(idxs[:last]):
            #Skip anything that does't actually have any overlap
            if overlap_bb[step] < 0.1:
                overlap.append(0)
                continue

            path0 = center_lines[i]
            path1 = center_lines[j]

            path = np.concatenate([path0, path1[::-1]])
            path = [[int(x[0]), int(x[1])] for x in path]

            expected_scale = (scales[i] + scales[j])/2.0
            one_off_area = expected_scale**2 * (path0.shape[0] + path1.shape[0])/2.0

            simple_path = pyclipper.SimplifyPolygon(path, pyclipper.PFT_NONZERO)
            inter_area = 0
            for path in simple_path:
                inter_area += abs(pyclipper.Area(path))

            area_ratio  = inter_area / one_off_area
            area_ratio = 1.0 - area_ratio

            overlap.append(area_ratio)

        overlap = np.array(overlap)
        to_delete = np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        idxs = np.delete(idxs, to_delete)

    return pick
