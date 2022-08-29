"""
Mask R-CNN
Configurations and data loading code for Uranovet dataset.

Written by Melvin Gelbard, based on the work of Waleed Abdulla

----------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python uranovet.py train --dataset=../../../../data/Uranovet/single/ios/single_ios_1 --weights=imagenet

    # Train a new model starting from specific weights file
    python uranovet.py train --dataset=/path/to/dataset --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python uranovet.py train --dataset=/path/to/dataset --weights=last

    # Generate submission file
    python uranovet.py detect --dataset=/path/to/dataset --weights=<last or /path/to/weights.h5>
"""

import os
import sys
import math
import random
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import json
import skimage
import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend as k

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/uranovet_single/")

print("Using TensorFlow: " + tf.__version__)


class UranovetConfig(Config):
    """Configuration for training on the Uranovet dataset.
    Derives from the base Config class and overrides values specific
    to the Uranovet dataset.
    """
    # Give the configuration a recognizable name
    NAME = "uranovet_single"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cassette

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8
    
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 10
    


class UranovetDataset(utils.Dataset):
    """
    Loads the Uranovet dataset from local machine
    """
    def load_uranovet(self, dataset_dir, count):
        """Generate the requested number of uranovet images.
        count: number of images to load.
        """
        # Add classes. We have only one class to add.
        self.add_class("cassette", 1, "cassette")

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "antigen_single_annotations.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "cassette",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cassette dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cassette":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cassette":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
        """Train the model."""
        # Training dataset.
        dataset_train = UranovetDataset()
        dataset_train.load_uranovet(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = UranovetDataset()
        dataset_val.load_uranovet(args.dataset, "val")
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***
        # Since we're using a very small dataset, and starting from
        # COCO trained weights, we don't need to train too long. Also,
        # no need to train all layers, just the heads should do it.
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=60,
                    layers='heads')


def detect(model, image_path=None, directory_path=None):
    assert image_path or directory_path

    if directory_path:
        all_images = [f for f in listdir(directory_path) if isfile(join(directory_path, f)) and f.endswith(".png")]
    else:
        all_images = [image_path]

    for image_path in all_images:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(join(directory_path, image_path))[:, :, :3]
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Save output
        file_name = directory_path + "/result_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        
        
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            ["Background", "Cassette"], r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions", show_plot=False)
        plt.savefig(file_name)
        print(image_path + " is saved in " + file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Uranovet cassettes.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'validate' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/uranovet/dataset/",
                        help='Directory of the Uranovet dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to detect cassette from')
    parser.add_argument('--directory', required=False,
                        metavar="path to directory of images",
                        help='directory of imagesto detect cassette from')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        if not args.dataset:
            args.dataset = os.getcwd()
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image or args.directory,\
               "Provide --image or --directory to apply detection"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = UranovetConfig()
    else:
        class InferenceConfig(UranovetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model, image_path=args.image, directory_path=args.directory)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
