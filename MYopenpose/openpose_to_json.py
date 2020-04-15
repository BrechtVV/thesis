import json

def parse_json(input_path, output_path):    
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