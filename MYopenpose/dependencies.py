import sys
import cv2
import os
from sys import platform
import argparse

def import_openpose(build_path):
    try:
        # Import Openpose (Ubuntu)
        try:
            sys.path.append(build_path + 'python')
            from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e
    except Exception as e:
        print(e)
        sys.exit(-1)
