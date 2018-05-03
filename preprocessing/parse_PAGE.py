import sys
import os
from os import path, listdir
from os.path import join, isfile
from collections import defaultdict
import xml.etree.ElementTree
import re
import json

def extract_points(data_string):
    return [tuple(int(x) for x in v.split(',')) for v in data_string.split()]

# http://stackoverflow.com/a/12946675/3479446
def get_namespace(element):
    m = re.match('\{.*\}', element.tag)
    return m.group(0) if m else ''

def readXMLFile(xml_string):
    root = xml.etree.ElementTree.fromstring(xml_string)
    namespace = get_namespace(root)

    return processXML(root, namespace)

def processXML(root, namespace):
    pages = []
    for page in root.findall(namespace+'Page'):
        pages.append(process_page(page, namespace))

    return pages

def process_page(page, namespace):

    page_out = {}
    regions = []
    lines = []
    for region in page.findall(namespace+'TextRegion'):
        region_out, region_lines = process_region(region, namespace)

        regions.append(region_out)
        lines += region_lines

    graphic_regions = []
    for region in page.findall(namespace+'GraphicRegion'):
        region_out, region_lines = process_region(region, namespace)
        graphic_regions.append(region_out)

    page_out['regions'] = regions
    page_out['lines'] = lines
    page_out['graphic_regions'] = graphic_regions

    return page_out

def process_region(region, namespace):

    region_out = {}

    coords = region.find(namespace+'Coords')
    region_out['bounding_poly'] = extract_points(coords.attrib['points'])
    region_out['id'] = region.attrib['id']

    lines = []
    for line in region.findall(namespace+'TextLine'):
        line_out = process_line(line, namespace)
        line_out['region_id'] = region.attrib['id']
        lines.append(line_out)

    ground_truth = None
    text_equiv = region.find(namespace+'TextEquiv')
    if text_equiv is not None:
        ground_truth = text_equiv.find(namespace+'Unicode').text

    region_out['ground_truth'] = ground_truth

    return region_out, lines

def process_line(line, namespace):
    errors = []
    line_out = {}

    if 'custom' in line.attrib:
        custom = line.attrib['custom']
        custom = custom.split(" ")
        if "readingOrder" in custom:
            roIdx = custom.index("readingOrder")
            ro = int("".join([v for v in custom[roIdx+1] if v.isdigit()]))
            line_out['read_order'] = ro

    if 'id' in line.attrib:
        line_out['id'] = line.attrib['id']

    baseline = line.find(namespace+'Baseline')

    if baseline is not None:
        line_out['baseline'] = extract_points(baseline.attrib['points'])
    else:
        errors.append('No baseline')

    coords = line.find(namespace+'Coords')
    line_out['bounding_poly'] = extract_points(coords.attrib['points'])

    ground_truth = None
    text_equiv = line.find(namespace+'TextEquiv')
    if text_equiv is not None:
        ground_truth = text_equiv.find(namespace+'Unicode').text

    if ground_truth == None or len(ground_truth) == 0:
        errors.append("No ground truth")
        ground_truth = ""

    line_out['ground_truth'] = ground_truth
    if len(errors) > 0:
        line_out['errors'] = errors

    return line_out

    return {"images":images}

if __name__ == '__main__':
    xml_file = sys.argv[1]
    image_file = sys.argv[2]
    xmlFileResult = readXMLFile(xml_file)
