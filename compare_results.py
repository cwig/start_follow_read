import sys
import os
from preprocessing import parse_PAGE
from collections import defaultdict
import editdistance

def read_xml(filename):
    with open(filename) as f:
        xml_string_data = f.read()
    # xml_string_data = f1.replace("&amp;", "&")
    # xml_string_data = f1.replace("&", "&amp;")

    return xml_string_data

def get_lines_in_region(data):
    regions = defaultdict(list)
    for l in data['lines']:
        regions[l['region_id']].append(l)
    return regions

if __name__ == "__main__":

    f1 = sys.argv[1]
    f2 = sys.argv[2]

    f1_files = {}
    for root, folders, files in os.walk(f1):
        for f in files:
            if f.endswith(".xml"):
                f1_files[f] = os.path.join(root, f)

    f2_files = {}
    for root, folders, files in os.walk(f2):
        for f in files:
            if f.endswith(".xml"):
                f2_files[f] = os.path.join(root, f)
    print len(f1_files)
    print len(f2_files)

    sum_dif = 0
    results =[]
    running_sum = 0

    for i, k in enumerate(sorted(f1_files)):
        filename = k
        f1 = f1_files[k]
        f2 = f2_files[k]

        xml1 = read_xml(f1)
        xml2 = read_xml(f2)

        data1 = parse_PAGE.readXMLFile(xml1)[0]
        data2 = parse_PAGE.readXMLFile(xml2)[0]


        region1 = get_lines_in_region(data1)
        region2 = get_lines_in_region(data2)

        joint_set = set(region1.keys()) | set(region2.keys())
        xor_set = set(region1.keys()) ^ set(region2.keys())

        if len(xor_set) != 0:
            print k, xor_set

        for k in set(region1.keys()) | set(region2.keys()):

            full_r1 = "\n".join([l['ground_truth'] for l in region1[k] ])
            full_r2 = "\n".join([l['ground_truth'] for l in region2[k] ])

            dis = editdistance.eval(full_r1, full_r2)

            length = (len(full_r1) + len(full_r2))
            if length == 0:
                out = 0
            else:
                out = dis / float(length)

            results.append((out, filename, k, i, full_r1, full_r2))
            sum_dif += out
    print "WER", sum_dif
