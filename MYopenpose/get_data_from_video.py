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
parser.add_argument("--image_path", default="media/example1.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--output", default="output/", help="Output folder.")
args = parser.parse_known_args()

output_folder = args[0].output
json_path = output_folder + "json/"
if not os.path.exists(json_path):
    os.makedirs(json_path)


# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = models_path
params["heatmaps_add_parts"] = True
params["heatmaps_add_bkg"] = True
params["heatmaps_add_PAFs"] = True
params["heatmaps_scale"] = 2
params["write_json"] = json_path

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()



cap = cv2.VideoCapture(args[0].image_path)
i = 0
while True:
    # read frames   
    ret, img = cap.read()
    if img is None:
        break
    frame_folder = output_folder + "frame" + str(i).zfill(6) + "/"
    heatmap_path = frame_folder + "heatmap/"
    os.makedirs(heatmap_path)
    processImage(opWrapper, img, frame_folder + "skeleton.jpg", heatmap_path)
    i+=1
    #skip some frames
    #for x in range(10):
    #    ret, img = cap.read()
    #    i+=1

cap.release()



