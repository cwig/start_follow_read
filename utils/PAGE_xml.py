import re
from preprocessing import parse_PAGE
import xml.etree.ElementTree as ET
from e2e import e2e_postprocessing
import pyclipper
import numpy as np
ET.register_namespace("","http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15")

# http://stackoverflow.com/a/12946675/3479446
def get_namespace(element):
    m = re.match('\{.*\}', element.tag)
    return m.group(0) if m else ''


def load_xml(xml_file):
    with open(xml_file) as f:
        xml_data = f.read()

    #the xml files have improper & escaping
    xml_data = xml_data.replace("&amp;", "&")
    xml_data = xml_data.replace("&", "&amp;")
    root = ET.fromstring(xml_data)
    tree = ET.ElementTree(root)

    namespace = get_namespace(root)

    return tree, root, namespace


def create_output_xml(xml_path, out, output_strings, output_path):
        tree, root, namespace = load_xml(xml_path)
        data = parse_PAGE.processXML(root, namespace)


        scores, background_scores, total_areas = assign_path_to_region(out, data[0]['regions'])
        all_scores = np.concatenate([background_scores[None,:], scores])
        best_match = np.argmax(all_scores, axis=0)-1

        region_ids = [r['id'] for r in data[0]['regions']]

        empty_regions(root)
        load_regions(root, region_ids, best_match, out, output_strings)

        tree.write(output_path, encoding="UTF-8",xml_declaration=True)

def create_output_xml_roi(xml_path, out, output_strings, output_path, roi_id):
    tree, root, namespace = load_xml(xml_path)
    data = parse_PAGE.processXML(root, namespace)

    fill_region(root, {
        roi_id: (out, output_strings)
    })

    tree.write(output_path, encoding="UTF-8",xml_declaration=True)

def empty_regions(root):
    namespace = get_namespace(root)
    for page in root.findall(namespace+'Page'):
        for region in page.findall(namespace+"TextRegion"):
            for text_line in list(region.findall(namespace+"TextLine")):
                region.remove(text_line)

def assign_path_to_region(out, regions):

    trimmed_polys = e2e_postprocessing.get_trimmed_polygons(out)

    polys = []
    for t in trimmed_polys:
        p = t[:,:2,0].tolist() + t[::-1,:2,1].tolist()
        polys.append(p)

    scores = []
    for i, r in enumerate(regions):
        region_poly = r['bounding_poly']
        scores_i = []
        scores.append(scores_i)
        for p in polys:
            pc = pyclipper.Pyclipper()

            try:
                pc.AddPath(p, pyclipper.PT_CLIP, True)
            except:
                scores_i.append(0)
                # print p
                print "Failed to assign text line, probably not an issue"
                continue
            pc.AddPath(region_poly, pyclipper.PT_SUBJECT, True)

            solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

            # pts = np.array(region_poly, np.int32)
            # pts = pts.reshape((-1,1,2))
            # cv2.polylines(img,[pts],True,(0,0,255), thickness=3)

            area = 0
            for path in solution:
                area += pyclipper.Area(path)

            scores_i.append(area)

    background_scores = []
    total_areas = []
    for p in polys:
        pc = pyclipper.Pyclipper()
        try:
            pc.AddPath(p, pyclipper.PT_CLIP, True)
        except:
            total_areas.append(np.inf)
            background_scores.append(np.inf)
            # print p
            print "Failed to assign text line, probably not an issue"
            continue
        pc.AddPaths([r['bounding_poly'] for r in regions], pyclipper.PT_SUBJECT, True)
        solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        area = 0
        for path in solution:
            area += pyclipper.Area(path)

        simple_path = pyclipper.SimplifyPolygon(p, pyclipper.PFT_NONZERO)
        total_area = 0
        for path in simple_path:
            total_area += pyclipper.Area(path)

        total_areas.append(total_area)
        background_score = total_area - area
        background_scores.append(background_score)

    return np.array(scores), np.array(background_scores), np.array(total_areas)

def fill_region(root, region_id_map):
    namespace = get_namespace(root)

    cnt = 0
    for page in root.findall(namespace+'Page'):
        for region in page.findall(namespace+"TextRegion"):
            region_id = region.attrib['id']

            if region_id not in region_id_map:
                continue

            out_pick, output_strings_pick = region_id_map[region_id]

            for i, handwriting_text in enumerate(output_strings_pick):
                text_line = ET.Element(namespace+"TextLine")
                region.append(text_line)
                text_line.attrib['reference_image_path'] = "line_" + str(i) +"_"+region_id + ".png"
                # text_line.attrib['id'] = "line_" + str(i) +"_"+region_id

                line_coords = ET.Element(namespace+"Coords")
                coord_str = "{},{} {},{}".format(cnt, cnt, cnt+1, cnt+1)
                line_coords.attrib['points'] = coord_str
                text_line.append(line_coords)

                line_baseline = ET.Element(namespace+"Baseline")
                line_baseline.attrib['points'] = coord_str
                text_line.append(line_baseline)
                cnt += 2

                text_equiv = ET.Element(namespace+"TextEquiv")
                text_line.append(text_equiv)

                line_unicode = ET.Element(namespace+"Unicode")
                text_equiv.append(line_unicode)
                line_unicode.text = handwriting_text

def load_regions(root, region_ids, alignments, out, output_strings):
    namespace = get_namespace(root)

    region_id_map = {}
    for page in root.findall(namespace+'Page'):
        for region in page.findall(namespace+"TextRegion"):
            region_id = region.attrib['id']
            region_id_idx = region_ids.index(region_id)
            out_select = np.where(alignments == region_id_idx)
            out_pick = e2e_postprocessing.filter_on_pick_no_copy(out, out_select)
            output_strings_pick = [output_strings[i] for i in out_select[0]]
            region_id_map[region_id] = (out_pick, output_strings_pick)

    fill_region(root, region_id_map)
