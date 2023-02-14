from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

import os
import pickle

from utils import initial_predict_config

DIR_OUT = './predict'

os.makedirs(DIR_OUT, exist_ok=True)

import cv2

(
    dataset_name, 
    images_path, 
    dataset_json_annotations_path, 
    path, 
    iou, 
    cfg_save_path,
) = initial_predict_config()

register_coco_instances(
    name=dataset_name, 
    metadata={}, 
    json_file=dataset_json_annotations_path, 
    image_root=images_path
)

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

with open("./output/last_checkpoint", "r") as arquivo:
    last_checkpoint = arquivo.read()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, last_checkpoint)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = iou
cfg.DATASETS.TEST = (dataset_name, )

dataset_custom = DatasetCatalog.get(dataset_name)
dataset_custom_metadata = MetadataCatalog.get(dataset_name)

predictor = DefaultPredictor(cfg)

file_name = path.split('/')[-1]

img = cv2.imread(path)

outputs = predictor(img)

v = Visualizer(img[:, :, ::-1],
    metadata=dataset_custom_metadata, 
    scale=1,
    instance_mode=ColorMode.IMAGE   
)

v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imwrite(f"{DIR_OUT}/predict-{file_name}.png", v.get_image()[:, :, ::-1])