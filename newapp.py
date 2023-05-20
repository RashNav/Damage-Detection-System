
import argparse
import io
from PIL import Image
import datetime
import numpy as np
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response, jsonify, flash
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import warnings
warnings.filterwarnings('ignore')
import sys
import json
import skimage.draw
import cv2
import random
import math
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils




# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Root directory of the project
ROOT_DIR = "/home/rash/Desktop/WEB_APPLICATION"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"


    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + scratches, dents

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7


class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.98

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config,model_dir=DEFAULT_LOGS_DIR)


################################################################################

UPLOAD_FOLDER = '/home/rash/Desktop/WEB_APPLICATION/static/images/'

app = Flask(__name__,static_folder="static")
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
                # Load the image
                image_data = request.files["file"].read()
                image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                #inference_config = InferenceConfig()
                
                #model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=DEFAULT_LOGS_DIR)
                model_path = '/home/rash/Desktop/WEB_APPLICATION/model/mask_rcnn_object_0003.h5'
                
                # Load trained weights
                print("Loading weights from ", model_path)
                model.load_weights(model_path, by_name=True)
                
                # Run inference
                results = model.detect([image], verbose=1)
                
                # Visualize the results
                r = results[0]

                class_names = ['BG', 'scratch', 'dent']
                result = visualize.display_instances(
                    image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figsize=(5,5,), save_fig_path='/home/rash/Desktop/WEB_APPLICATION/static/images/after1.jpg'
                )
                image_filename = 'after1.jpg'

                return render_template('index.html',processed_image=image_filename)
            
    except Exception as e:
        # Handle the exception and provide an appropriate error message
        return f"An error occurred: {str(e)}"
                               
if __name__ == "__main__":
    app.run()
