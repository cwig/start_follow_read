import sys
import parse_PAGE
import cv2
import line_extraction
import numpy as np
import os
import traceback
from collections import defaultdict
from scipy import ndimage
import json
import codecs


def handle_single_image(xml_path, img_path, output_directory, config={}):

    output_data = []

    with open(xml_path) as f:
        num_lines = sum(1 for line in f.readlines() if len(line.strip())>0)

    if num_lines > 0:

        all_lines = ""

        with open(xml_path) as f:
            xml_string_data = f.read()

        #Parse invalid xml data
        xml_string_data = xml_string_data.replace("&amp;", "&")
        xml_string_data = xml_string_data.replace("&", "&amp;")

        xml_data = parse_PAGE.readXMLFile(xml_string_data)

        basename = xml_path.split("/")[-1][:-len(".xml")]


        if len(xml_data) > 1:
            raise Exception("Not handling this correctly")

        for region in xml_data[0]['regions']:
            region_output_data = []

            region_lines =  region['ground_truth'].split("\n")
            region_lines = [s.strip() for s in region_lines]
            region_lines = [s for s in region_lines if len(s) > 0]

            for line in region_lines:
                output_data.append({
                    "gt": line
                })

    else:
        print "WARNING: {} has no lines".format(xml_path)

    output_data_path =os.path.join(output_directory, basename, "{}.json".format(basename))
    if not os.path.exists(os.path.dirname(output_data_path)):
        os.makedirs(os.path.dirname(output_data_path))

    with open(output_data_path, 'w') as f:
        json.dump(output_data, f)

    return output_data_path

def find_best_xml(list_of_files, filename):

    if len(list_of_files) <= 1:
        return list_of_files

    print "Selecting multiple options from:"

    line_cnts = []
    for xml_path in list_of_files:
        test_xml_path = os.path.join(xml_path, filename+".xml")
        print test_xml_path
        with open(test_xml_path) as f:
            num_lines = sum(1 for line in f.readlines() if len(line.strip())>0)
        line_cnts.append((num_lines, xml_path))
    line_cnts.sort(key=lambda x:x[0], reverse=True)
    print "Sorted by line count..."
    ret = [l[1] for l in line_cnts]
    return ret

def process_dir(xml_directory, img_directory, output_directory):
    xml_filename_to_fullpath = defaultdict(list)
    for root, sub_folders, files in os.walk(xml_directory):
        for f in files:
            if not f.endswith(".xml"):
                continue
            f = f[:-len(".xml")]
            if f in xml_filename_to_fullpath:
                print "Error: this assumes no repeating files names: {} xml".format(f)

            xml_filename_to_fullpath[f].append(root)

    png_filename_to_fullpath = {}
    image_ext = {}
    for root, sub_folders, files in os.walk(img_directory):
        for f in files:
            valid_image_extensions = ['.jpg', '.png', '.JPG', '.PNG']
            if not any([f.endswith(v) for v in valid_image_extensions]):
                continue

            if f in png_filename_to_fullpath:
                print "Error: this assumes no repeating files names: {} img".format(f)

            extension = f[-len(".png"):]
            f = f[:-len(".png")]
            image_ext[f] = extension
            png_filename_to_fullpath[f] = root

    xml_not_imgs = set(xml_filename_to_fullpath.keys()) - set(png_filename_to_fullpath.keys())
    print "Files in XML but not Images", len(xml_not_imgs)
    # if len(xml_not_imgs) > 0:
    #     print xml_not_imgs
    img_not_xml = set(png_filename_to_fullpath.keys()) - set(xml_filename_to_fullpath.keys())
    print "Files in Images but not XML", len(img_not_xml)
    # if len(img_not_xml) > 0:
    #     print img_not_xml
    print ""
    to_process = set(xml_filename_to_fullpath.keys()) & set(png_filename_to_fullpath.keys())
    print "Number to be processed", len(to_process)

    all_ground_truth = []
    for i, filename in enumerate(list(to_process)):
        if i%1000==0:
            print i
        img_path = png_filename_to_fullpath[filename]
        xml_paths = xml_filename_to_fullpath[filename]

        out_rel = os.path.relpath(img_path, img_directory)
        this_output_directory = os.path.join(output_directory, out_rel)

        img_path = os.path.join(img_path, filename+image_ext[filename])
        success = False
        xml_path = find_best_xml(xml_paths, filename)[0]

        xml_path = os.path.join(xml_path, filename+".xml")

        json_path = handle_single_image(xml_path, img_path, this_output_directory, {})
        all_ground_truth.append([json_path, img_path])

    return all_ground_truth

if __name__ == "__main__":
    xml_directory = sys.argv[1]
    img_directory = sys.argv[2]

    output_directory = sys.argv[3]
    training_output_json = sys.argv[4]
    validation_output_json = sys.argv[5]
    all_ground_truth = process_dir(xml_directory, img_directory, output_directory)
    all_ground_truth.sort()


    training_list = all_ground_truth[:9000]
    validation_list = all_ground_truth[9000:]

    print "Training Size:", len(training_list)
    print "Validation Size:", len(validation_list)

    with open(training_output_json, 'w') as f:
        json.dump(training_list, f)

    with open(validation_output_json, 'w') as f:
        json.dump(validation_list, f)
