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
from svgpathtools import Path, Line
from scipy.interpolate import griddata

def generate_offset_mapping(img, ts, path, offset_1, offset_2, max_min = None, cube_size = None):
    # cube_size = 80

    offset_1_pts = []
    offset_2_pts = []
    # for t in ts:
    for i in range(len(ts)):
        t = ts[i]
        pt = path.point(t)

        norm = None
        if i == 0:
            norm = normal(pt, path.point(ts[i+1]))
            norm = norm / dis(complex(0,0), norm)
        elif i == len(ts)-1:
            norm = normal(path.point(ts[i-1]), pt)
            norm = norm / dis(complex(0,0), norm)
        else:
            norm1 = normal(path.point(ts[i-1]), pt)
            norm1 = norm1 / dis(complex(0,0), norm1)
            norm2 = normal(pt, path.point(ts[i+1]))
            norm2 = norm2 / dis(complex(0,0), norm2)

            norm = (norm1 + norm2)/2
            norm = norm / dis(complex(0,0), norm)

        offset_vector1 = offset_1 * norm
        offset_vector2 = offset_2 * norm

        pt1 = pt + offset_vector1
        pt2 = pt + offset_vector2

        offset_1_pts.append(complexToNpPt(pt1))
        offset_2_pts.append(complexToNpPt(pt2))

    offset_1_pts = np.array(offset_1_pts)
    offset_2_pts = np.array(offset_2_pts)

    h,w = img.shape[:2]

    offset_source2 = np.array([(cube_size*i, 0) for i in range(len(offset_1_pts))], dtype=np.float32)
    offset_source1 = np.array([(cube_size*i, cube_size) for i in range(len(offset_2_pts))], dtype=np.float32)

    offset_source1 = offset_source1[::-1]
    offset_source2 = offset_source2[::-1]

    source = np.concatenate([offset_source1, offset_source2])
    destination = np.concatenate([offset_1_pts, offset_2_pts])

    source = source[:,::-1]
    destination = destination[:,::-1]

    n_w = int(offset_source2[:,0].max())
    n_h = int(cube_size)

    grid_x, grid_y = np.mgrid[0:n_h, 0:n_w]

    grid_z = griddata(source, destination, (grid_x, grid_y), method='cubic')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(n_h,n_w)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(n_h,n_w)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    rectified_to_warped_x = map_x_32
    rectified_to_warped_y = map_y_32

    grid_x, grid_y = np.mgrid[0:h, 0:w]
    grid_z = griddata(source, destination, (grid_x, grid_y), method='cubic')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(h,w)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(h,w)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    warped_to_rectified_x = map_x_32
    warped_to_rectified_y = map_y_32

    return rectified_to_warped_x, rectified_to_warped_y, warped_to_rectified_x, warped_to_rectified_y, max_min


def dis(pt1, pt2):
    a = (pt1.real - pt2.real)**2
    b = (pt1.imag - pt2.imag)**2
    return np.sqrt(a+b)

def complexToNpPt(pt):
    return np.array([pt.real, pt.imag], dtype=np.float32)

def normal(pt1, pt2):
    dif = pt1 - pt2
    return complex(-dif.imag, dif.real)

def find_t_spacing(path, cube_size):
    l = path.length()
    error = 0.01
    init_step_size = cube_size / l

    last_t = 0
    cur_t = 0
    pts = []
    ts = [0]
    pts.append(complexToNpPt(path.point(cur_t)))
    path_lookup = {}
    for target in np.arange(cube_size, int(l), cube_size):
        step_size = init_step_size
        for i in range(1000):
            cur_length = dis(path.point(last_t), path.point(cur_t))
            if np.abs(cur_length - cube_size) < error:
                break

            step_t = min(cur_t + step_size, 1.0)
            step_l = dis(path.point(last_t), path.point(step_t))

            if np.abs(step_l - cube_size) < np.abs(cur_length - cube_size):
                cur_t = step_t
                continue

            step_t = max(cur_t - step_size, 0.0)
            step_t = max(step_t, last_t)
            step_t = max(step_t, 1.0)

            step_l = dis(path.point(last_t), path.point(step_t))

            if np.abs(step_l - cube_size) < np.abs(cur_length - cube_size):
                cur_t = step_t
                continue

            step_size = step_size / 2.0

        last_t = cur_t

        ts.append(cur_t)
        pts.append(complexToNpPt(path.point(cur_t)))

    pts = np.array(pts)

    return ts

def handle_single_image(xml_path, img_path, output_directory, config={}):

    output_data = []

    with open(xml_path) as f:
        num_lines = sum(1 for line in f.readlines() if len(line.strip())>0)

    img = cv2.imread(img_path)
    if num_lines > 0:

        all_lines = ""
        # with codecs.open(xml_path, encoding='utf-8') as f:
        #     xml_string_data = f.read()

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
            region_mask = line_extraction.extract_region_mask(img, region['bounding_poly'])



            for i, line in enumerate(xml_data[0]['lines']):
                if line['region_id'] != region['id']:
                    continue

                if 'bounding_poly' not in line:
                    print "Warning: Missing bounding poly {}".format(xml_path)
                    print line
                    continue

                if 'baseline' not in line:
                    print "Warning: Missing baseline {}".format(xml_path)
                    print line
                    continue

                line_mask = line_extraction.extract_region_mask(img, line['bounding_poly'])

                masked_img = img.copy()
                masked_img[line_mask==0] = 0

                summed_axis0 = (masked_img.astype(float) / 255).sum(axis=0)
                summed_axis1 = (masked_img.astype(float) / 255).sum(axis=1)

                non_zero_cnt0 = np.count_nonzero(summed_axis0) / float(len(summed_axis0))
                non_zero_cnt1 = np.count_nonzero(summed_axis1) / float(len(summed_axis1))

                avg_height0 = np.median(summed_axis0[summed_axis0 != 0])
                avg_height1 = np.median(summed_axis1[summed_axis1 != 0])

                avg_height = min(avg_height0, avg_height1)
                if non_zero_cnt0 > non_zero_cnt1:
                    target_step_size = avg_height0
                else:
                    target_step_size = avg_height1

                paths = []
                for i in range(len(line['baseline'])-1):
                    i_1 = i+1

                    p1 = line['baseline'][i]
                    p2 = line['baseline'][i_1]

                    p1_c = complex(*p1)
                    p2_c = complex(*p2)


                    paths.append(Line(p1_c, p2_c))


                # Add a bit on the end
                tan = paths[-1].unit_tangent(1.0)
                p3_c = p2_c + target_step_size * tan
                paths.append(Line(p2_c, p3_c))

                path = Path(*paths)

                ts = find_t_spacing(path, target_step_size)

                #Changing this causes issues in pretraining - not sure why
                target_height = 32

                rectified_to_warped_x, rectified_to_warped_y, warped_to_rectified_x, warped_to_rectified_y, max_min = generate_offset_mapping(masked_img, ts, path, 0, -2*target_step_size, cube_size = target_height)
                warped_above = cv2.remap(line_mask, rectified_to_warped_x, rectified_to_warped_y, cv2.INTER_CUBIC, borderValue=(0,0,0))

                rectified_to_warped_x, rectified_to_warped_y, warped_to_rectified_x, warped_to_rectified_y, max_min = generate_offset_mapping(masked_img, ts, path, 2*target_step_size, 0, cube_size = target_height)
                warped_below = cv2.remap(line_mask, rectified_to_warped_x, rectified_to_warped_y, cv2.INTER_CUBIC, borderValue=(0,0,0))

                above_scale =  np.max((warped_above.astype(float) / 255).sum(axis=0))
                below_scale = np.max((warped_below.astype(float) / 255).sum(axis=0))

                ab_sum = above_scale + below_scale
                above = target_step_size * (above_scale/ab_sum)
                below = target_step_size * (below_scale/ab_sum)

                above = target_step_size * (above_scale/(target_height/2.0))
                below = target_step_size * (below_scale/(target_height/2.0))
                target_step_size = above + below
                ts = find_t_spacing(path, target_step_size)

                rectified_to_warped_x, rectified_to_warped_y, warped_to_rectified_x, warped_to_rectified_y, max_min = generate_offset_mapping(masked_img, ts, path, below, -above, cube_size=target_height)

                rectified_to_warped_x = rectified_to_warped_x[::-1,::-1]
                rectified_to_warped_y = rectified_to_warped_y[::-1,::-1]
                warped_to_rectified_x = warped_to_rectified_x[::-1,::-1]
                warped_to_rectified_y = warped_to_rectified_y[::-1,::-1]

                warped = cv2.remap(img, rectified_to_warped_x, rectified_to_warped_y, cv2.INTER_CUBIC, borderValue=(255,255,255))

                mapping = np.stack([rectified_to_warped_y, rectified_to_warped_x], axis=2)

                top_left = mapping[0,0,:] / np.array(img.shape[:2]).astype(np.float32)
                btm_right = mapping[min(mapping.shape[0]-1, target_height-1), min(mapping.shape[1]-1, target_height-1),:] / np.array(img.shape[:2]).astype(np.float32)


                line_points = []
                for i in xrange(0,mapping.shape[1],target_height):

                    x0 = float(rectified_to_warped_x[0,i])
                    x1 = float(rectified_to_warped_x[-1,i])

                    y0 = float(rectified_to_warped_y[0,i])
                    y1 = float(rectified_to_warped_y[-1,i])

                    line_points.append({
                        "x0": x0,
                        "x1": x1,
                        "y0": y0,
                        "y1": y1
                    })

                output_file = os.path.join(output_directory, basename, "{}~{}~{}.png".format(basename, line['region_id'], line['id']))
                warp_output_file = os.path.join(output_directory, basename, "{}-{}.png".format(basename, str(len(region_output_data))))
                warp_output_file_save = os.path.join(basename, "{}-{}.png".format(basename, str(len(region_output_data))))
                save_file = os.path.join(basename, "{}~{}~{}.png".format(basename, line['region_id'], line['id']))
                region_output_data.append({
                    "gt": line.get("ground_truth", ""),
                    "image_path": save_file,
                    "sol": line_points[0],
                    "lf": line_points,
                    "hw_path": warp_output_file_save
                })
                if not os.path.exists(os.path.dirname(output_file)):
                    try:
                        os.makedirs(os.path.dirname(output_file))
                    except OSError as exc:
                        raise Exception("Could not write file")

                cv2.imwrite(warp_output_file, warped)

            output_data.extend(region_output_data)

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
    if len(xml_not_imgs) > 0:
        print xml_not_imgs
    img_not_xml = set(png_filename_to_fullpath.keys()) - set(xml_filename_to_fullpath.keys())
    print "Files in Images but not XML", len(img_not_xml)
    if len(img_not_xml) > 0:
        print img_not_xml
    print ""
    to_process = set(xml_filename_to_fullpath.keys()) & set(png_filename_to_fullpath.keys())
    print "Number to be processed", len(to_process)

    all_ground_truth = []
    for i, filename in enumerate(list(to_process)):
        if i%1==0:
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

    training_list = all_ground_truth[:45]
    validation_list = all_ground_truth[45:]

    print "Training Size:", len(training_list)
    print "Validation Size:", len(validation_list)

    with open(training_output_json, 'w') as f:
        json.dump(training_list, f)

    with open(validation_output_json, 'w') as f:
        json.dump(validation_list, f)
