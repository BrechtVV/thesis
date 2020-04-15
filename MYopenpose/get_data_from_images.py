# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

build_path = "/content/openpose/build/"
models_path = "/content/openpose/models/"

from dependencies import *
import_openpose(build_path)
from openpose import pyopenpose as op
from process_image import *

# Flags
parser = argparse.ArgumentParser()
parser.add_argument('--images', type=str, default="./images")
parser.add_argument('--json-folder', type=str, default="./json")
parser.add_argument('--output-folder', type=str, default="./output")
#args = parser.parse_known_args()

args = parser.parse_args()

output_folder = args.output_folder

json_temp = os.path.join(args.json_folder, "temp")
if not os.path.exists(json_temp):
    os.makedirs(json_temp)


# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = models_path
params["heatmaps_add_parts"] = True
params["heatmaps_add_bkg"] = True
params["heatmaps_add_PAFs"] = True
params["heatmaps_scale"] = 2
params["write_json"] = json_temp

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

from openpose_to_json import *

index = 0
for f in os.listdir(args.images):
    image_path = os.path.join(args.images, f)
    image = cv2.imread(image_path)
    output_path = os.path.join(args.output_folder, f)
    process_image(opWrapper, image, output_path)

    json_input = os.path.join(json_temp, str(index) + "_keypoints.json") 
    json_output = os.path.join(args.json_folder, f[:-3] + ".json")
    parse_json(json_input, json_output)
    index += 1

import shutil
shutil.rmtree(json_temp)