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


def convert(humans, width, height, json_path="result.json"):
    output = {}
    output['people'] = []

    for human in humans:
        h = {}
        for idx, bp in human.body_parts.items():
            #print(idx, CocoPart(idx), bp.x*width, bp.y*height, bp.score)
            h[idx] = [bp.x*width, bp.y*height, bp.score]
        output['people'].append(h)
    
    with open(json_path, 'w') as outfile:
        json.dump(output, outfile)

