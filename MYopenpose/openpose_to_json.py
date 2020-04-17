import json
from enum import Enum

class CocoPart(Enum):
    Nose = 0 
    Neck = 1 
    RShoulder = 2 
    RElbow = 3 
    RWrist = 4 
    LShoulder = 5 
    LElbow = 6 
    LWrist = 7 
    MidHip = 8 
    RHip = 9 
    RKnee = 10
    RAnkle = 11
    LHip = 12
    LKnee = 13
    LAnkle = 14
    REye = 15
    LEye = 16
    REar = 17
    LEar = 18
    LBigToe = 19
    LSmallToe = 20
    LHeel = 21
    RBigToe = 22
    RSmallToe = 23
    RHeel = 24
    Background = 25

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
                kp = CocoPart(int(i/3)).name
                h[kp] = [x, y, c]
            output['people'].append(h)

        with open(output_path, 'w') as outfile:
            json.dump(output, outfile)