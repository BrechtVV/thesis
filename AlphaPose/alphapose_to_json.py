import json
import os

def convert_open(input_path, output_path):
    with open(input_path) as json_file:
        data = json.load(json_file)
        
        output = {}
        output['people'] = []

        for p in data['people']:
            h = {}
            kps = p['pose_keypoints_2d']
            for i in range(0, len(kps), 3):
                x = kps[i]
                y = kps[i+1]
                c = kps[i+2]
                h[int(i/3)] = [x, y, c]
            output['people'].append(h)

        with open(output_path, 'w') as outfile:
            json.dump(output, outfile)


def convert_coco(input_path, output_path):
    data = {}   
    with open(input_path) as json_file:
        data = json.load(json_file)
        
    output = {}
    for d in data:
        im_id = d['image_id']
        if im_id not in output.keys():
            output[im_id] = {}
            output[im_id]['people'] = []
        
        h = {}
        kps = d['keypoints']
        for i in range(0, len(kps), 3):
            x = kps[i]
            y = kps[i+1]
            c = kps[i+2]
            h[int(i/3)] = [x, y, c]
        output[im_id]['people'].append(h)

    for k in output.keys():        
        filename = os.path.join(output_path, k[:-3] + "json")
        out = output[k]
        with open(filename, 'w') as outfile:
            json.dump(out, outfile)