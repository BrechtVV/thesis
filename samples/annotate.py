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
img = None

def click(event, x, y, flas, param):
    global i, human, img

    if i < len(keypoints):
        if event == cv2.EVENT_MBUTTONDOWN:
            human[keypoints[i]] = [0, 0]
            print(0, 0)
            i += 1
            if i >= len(keypoints):
                print("DONE")
            else:
                print(keypoints[i], human[keypoints[i]], ":")    
        
        if event == cv2.EVENT_LBUTTONDOWN:
            human[keypoints[i]] = [x, y]
            print(x, y)
            i += 1
            if i >= len(keypoints):
                print("DONE")
            else:
                print(keypoints[i], human[keypoints[i]], ":")    

            img = cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("img", img) 



def annotate_image(image_path, json_path):
    global i, human, img
    i = 0
    human = {}
    for kp in keypoints:
        human[kp] = [0, 0]

    img = cv2.imread(image_path)

    with open(json_path) as json_file:
        data = json.load(json_file)
        for kp in data['people'][0]:
            human[kp] = data['people'][0][kp]
            img = cv2.circle(img, (human[kp][0], human[kp][1]), 3, (255, 0, 0), -1)
    
    print(keypoints[i], human[keypoints[i]], ":")    

    
    cv2.imshow("img", img)
    cv2.setMouseCallback("img", click)

    output = {}
    output['people'] = []
    
    key = cv2.waitKey(0)
    if key == 27: #ESC
        return "stop"
    elif key == ord('n'):
        return "next"
    
    output['people'].append(human)

    with open(json_path, 'w') as outfile:
        json.dump(output, outfile)



for f in sorted(os.listdir("images")):
    img_path = os.path.join("images", f)
    json_path = os.path.join("keypoints", f[:-4] + ".json")
    print(img_path, json_path)
    if annotate_image(img_path, json_path) == "stop":
        break