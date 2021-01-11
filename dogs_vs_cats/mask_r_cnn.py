#download cats_vs_dogs dataset from kaggle
import os
!kaggle competitions download -c dogs-vs-cats

#extract downloaded files
import zipfile
with zipfile.ZipFile("train.zip", 'r') as zip_ref:
  zip_ref.extractall()

#clone mask_rcnn repository from github
from git import Repo
Repo.clone_from("https://github.com/matterport/Mask_RCNN.git", "rcnn_master")

import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imsave
from PIL import Image
import tensorflow as tf
print(tf.__version__)

# Root directory of the project
ROOT_DIR = os.path.abspath("rcnn_master")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

#1.500 images per class
files = sorted(os.listdir("train"))
cats_batch = files[:1500]
dogs_batch = files[12500:14000]


def process_batch_npy(batch, name):
  object_masks = []
  for i, f in enumerate(batch):
    
    image = skimage.io.imread(os.path.join("train", f))
    
    results = model.detect([image], verbose=0)
    r = results[0]
    
    indices = [i for i, x in enumerate(r["class_ids"]) if x == class_names.index(name)]
    
    #merge masks if multiple objects
    mask = np.zeros(image.shape[0:2], dtype = "uint8")
    for i in indices:
      mask += r["masks"][:, :, i]

    #normalize to 0/1
    for i in range(mask.shape[0]):
      for j in range(mask.shape[1]):
        if mask[i, j] > 0:
          mask[i, j] = 1
    object_masks.append(mask * 255)
  np.save("{}_masks.npy".format(name), object_masks)
  return object_masks

dog_masks = process_batch_npy(dogs_batch, "dog")
cat_masks = process_batch_npy(catss_batch, "cat")