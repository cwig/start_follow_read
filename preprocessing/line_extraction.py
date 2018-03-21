import numpy as np
import cv2

def extract_region_mask(img, bounding_poly):
    pts = np.array(bounding_poly, np.int32)

    #http://stackoverflow.com/a/15343106/3479446
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    roi_corners = np.array([pts], dtype=np.int32)

    ignore_mask_color = (255,)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color, lineType=cv2.LINE_8)
    return mask

def extract_baseline(img, pts):
    new_pts = []
    for i in range(len(pts)-1):
        new_pts.append([pts[i], pts[i+1]])
    pts = np.array(new_pts, np.int32)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.polylines(mask,pts,False,255)
    return mask
