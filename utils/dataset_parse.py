import json
import os

def load_file_list(config):
    file_list_path = config['file_list']
    with open(file_list_path) as f:
        data = json.load(f)

    for d in data:
        json_path = os.path.join(config['json_folder'], d[0])
        img_path = os.path.join(config['img_folder'], d[1])

        d[0] = json_path
        d[1] = img_path

    return data
