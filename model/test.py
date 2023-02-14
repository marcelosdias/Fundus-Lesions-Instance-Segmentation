from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor

from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetCatalog, MetadataCatalog

from CocoEvaluator import CocoEvaluator

import os
import pickle

from utils import *

dataset_name, dataset_path, dataset_json_annotations_path, iou, cfg_save_path = initial_test_config()

register_coco_instances(
    name=dataset_name, 
    metadata={}, 
    json_file=dataset_json_annotations_path, 
    image_root=dataset_path
)

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

with open('./output/last_checkpoint', 'r') as arquivo:
    last_checkpoint = arquivo.read()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, last_checkpoint)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = iou

cfg.DATASETS.TEST = (dataset_name, )
    
dataset_custom = DatasetCatalog.get(dataset_name)
dataset_custom_metadata = MetadataCatalog.get(dataset_name)

predictor = DefaultPredictor(cfg)

dataset_custom = DatasetCatalog.get(dataset_name)
dataset_custom_metadata = MetadataCatalog.get(dataset_name)

evaluator = CocoEvaluator(dataset_name, cfg, False, output_dir='./output')

val_loader = build_detection_test_loader(cfg, dataset_name)

inference_on_dataset(predictor.model, val_loader, evaluator)