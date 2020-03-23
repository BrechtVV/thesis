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

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="media/example1.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--output", default="output/", help="Output folder.")
args = parser.parse_known_args()

outputFolder = args[0].output

json_path = outputFolder + "json"
if not os.path.exists(json_path):
    os.makedirs(json_path)
heatmap_path = outputFolder + "heatmap"
if not os.path.exists(heatmap_path):
    os.makedirs(heatmap_path)


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

# Process Image
datum = op.Datum()
imageToProcess = cv2.imread(args[0].image_path)
cv2.imwrite(outputFolder + "original.jpg", imageToProcess)
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop([datum])

# Display Image
#print("Body keypoints: \n" + str(datum.poseKeypoints))
cv2.imwrite(outputFolder + "skeleton.jpg", datum.cvOutputData)

# Process outputs
outputImageF = (datum.inputNetData[0].copy())[0,:,:,:] + 0.5
outputImageF = cv2.merge([outputImageF[0,:,:], outputImageF[1,:,:], outputImageF[2,:,:]])
outputImageF = (outputImageF*255.).astype(dtype='uint8')
heatmaps = datum.poseHeatMaps.copy()
heatmaps = (heatmaps).astype(dtype='uint8')

# Display Image
num_maps = heatmaps.shape[0]
for counter in range(num_maps):
    heatmap = heatmaps[counter, :, :].copy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
    cv2.imwrite(outputFolder + "heatmap_"+ str(counter).zfill(2) + ".jpg", heatmap)