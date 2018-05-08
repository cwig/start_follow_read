import sys
import os
from os import path, listdir
from os.path import join, isfile
from collections import defaultdict
import xml.etree.ElementTree as ET
import re
import json
import parse_PAGE
import numpy as np

import cv2

ET.register_namespace("","http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15")

import pyclipper
from copy import deepcopy

# http://stackoverflow.com/a/12946675/3479446
def get_namespace(element):
    m = re.match('\{.*\}', element.tag)
    return m.group(0) if m else ''

def load_xml(xml_file):
    with open(xml_file) as f:
        xml_data = f.read()

    xml_data = xml_data.replace("&amp;", "&")
    xml_data = xml_data.replace("&", "&amp;")
    root = ET.fromstring(xml_data)
    tree = ET.ElementTree(root)

    namespace = get_namespace(root)

    return tree, root, namespace

def create_mask_images(xml_path, original_img_path, output_path, image_id):
        tree, root, namespace = load_xml(xml_path)
        data = parse_PAGE.processXML(root, namespace)
        img = cv2.imread(original_img_path)
        median_color = np.median(img, axis=(0,1))

        print "Processing Image", image_id

        for i, r in enumerate(data[0]['regions']):
            region_poly = r['bounding_poly']
            print r['id']

            pts = np.array(region_poly, np.int32)
            #http://stackoverflow.com/a/15343106/3479446
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            roi_corners = np.array([pts], dtype=np.int32)

            ignore_mask_color = (255,)
            cv2.fillPoly(mask, roi_corners, ignore_mask_color, lineType=cv2.LINE_8)

            median_img = img.copy()
            median_img[mask == 0] = median_color

            output_dir = os.path.join(output_path, image_id)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            save_path = os.path.join(output_dir, image_id+"_"+r['id']+".png")
            cv2.imwrite(save_path, median_img)


if __name__ == "__main__":
    xml_folder = sys.argv[1]
    original_img_path = sys.argv[2]
    out_folder = sys.argv[3]

    xml_files = {}
    xml_out_files = {}
    for root, folders, files in os.walk(xml_folder):
        for f in files:
            if f.endswith(".xml"):
                basename = os.path.basename(f).replace(".xml", "")
                xml_files[basename] = os.path.join(root, f)
                xml_out_files[basename] = os.path.join(out_folder, basename+".png")

    image_files = {}
    for root, folders, files in os.walk(original_img_path):
        for f in files:
            if f.endswith(".jpg"):
                basename = os.path.basename(f).replace(".jpg", "")
                image_files[basename] = os.path.join(root, f)


    for k in xml_files:
        create_mask_images(xml_files[k], image_files[k], out_folder, k)
