import argparse
import logging
import sys
import time
import os

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def process_image(image_path, resize_out_ratio, json_folder, output_folder):
    # estimate human poses from a single image !
    image = common.read_imgfile(image_path, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % image_path)
        sys.exit(-1)

    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
    elapsed = time.time() - t

    #logger.info('inference image: %s in %.4f seconds.' % (image_path, elapsed))

    height, width = image.shape[:2]
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    
    b_name = os.path.basename(image_path)
    
    cv2.imwrite(os.path.join(output_folder,b_name), image)
    from humans_to_json import convert
    convert(humans, width, height, os.path.join(json_folder, b_name[:-3] + ".json"))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--images', type=str, default="./images")
    parser.add_argument('--json-folder', type=str, default="./json")
    parser.add_argument('--output-folder', type=str, default="./output")
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))


    for f in os.listdir(args.images):
        process_image(os.path.join(args.images,f), args.resize_out_ratio, args.json_folder, args.output_folder)    



