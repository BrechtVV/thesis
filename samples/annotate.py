import cv2
import os
import json

keypoints = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle ","LHip ","LKnee ","LAnkle ","REye ","LEye ","REar ","LEar"]

## for image in images
##      for keypoint in keypoints
##          select keypoint
##          save keypoint
##      save keypoints for image

i = 0
human = {}


def click(event, x, y, flas, param):
    global i, human
    
    if event == cv2.EVENT_LBUTTONDOWN:
        human[keypoints[i]] = [x, y]
        print(keypoints[i], x, y)
        i += 1
        if i > len(keypoints):
            print("DONE")
        else:
            print(keypoints[i], ":")



def annotate_image(image_path, json_path):
    global i, human
    i = 0
    human = {}
    print(keypoints[i], ":")

    img = cv2.imread(image_path)
    cv2.imshow("img", img)
    cv2.setMouseCallback("img", click)

    output = {}
    output['people'] = []
    
    cv2.waitKey(0)
    
    output['people'].append(human)

    with open(json_path, 'w') as outfile:
        json.dump(output, outfile)



for f in sorted(os.listdir("images")):
    img_path = os.path.join("images", f)
    json_path = os.path.join("keypoints", f[:-4] + ".json")
    print(img_path, json_path)
    annotate_image(img_path, json_path)