import argparse
import json
import os
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
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

def convert_open(input_json, output_dir):
    with open(input_json) as json_file:
        data = json.load(json_file)
        for d in data[:]:
            i = 0
            output_path = os.path.join(output_dir, str(d['image_id'].split(".")[0]).zfill(10) + "_" + str(i).zfill(3) + ".json")
            while os.path.exists(output_path):
                i+=1
                output_path = os.path.join(output_dir, str(d['image_id'].split(".")[0]).zfill(10) + "_" + str(i).zfill(3) + ".json")

            convert_one(d['keypoints'], output_path)
        
def convert_one(kps, output_path):
    output = {}
    output['people'] = []
    h = {}
    for i in range(0, len(kps), 3):
        x = kps[i]
        y = kps[i+1]
        c = kps[i+2]
        kp = CocoPart(int(i/3)).name
        h[kp] = [x, y, c]
    output['people'].append(h)
    with open(output_path, 'w') as outfile:
        json.dump(output, outfile)

parser = argparse.ArgumentParser(description='AlphaPose Convert')
parser.add_argument('--input-json', help='input-directory')
parser.add_argument('--output-dir', help='output-directory')
args = parser.parse_args()

convert_open(args.input_json, args.output_dir)