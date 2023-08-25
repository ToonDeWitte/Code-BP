from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.visualizer import ColorMode

import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        print(s["file_name"])
        img = cv2.imread(s["file_name"])
        img = img[:,:,::-1]
        v = Visualizer(img, metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 6

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg


def on_image(image_path, predictor):
    img = cv2.imread(image_path)
    img = img[:,:,::-1]
    outputs = predictor(img)
    v = Visualizer(img, metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # masks = outputs["instances"].pred_masks.cpu().numpy()
    # boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

    # masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))[0]

    # print(masks)

    plt.figure(figsize=(15,20))
    plt.imshow(v.get_image())
    # plt.imshow(masks)
    plt.show()

def save_masks(image_path, predictor):
    img = cv2.imread(image_path)
    img = img[:,:,::-1]
    outputs = predictor(img)
    v = Visualizer(img, metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)

    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # masks = outputs["instances"].pred_masks.cpu().numpy()
    # boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

    masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))

    os.makedirs("./masks_{}".format(image_path.split("\\")[-1].split(".")[0]), exist_ok=True)

    for idx, mask in enumerate(masks):
        cv2.imwrite("./masks_{}/{}.png".format(image_path.split("\\")[-1].split(".")[0], idx), mask * 255)

    