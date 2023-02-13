from detectron2.data.datasets import register_coco_instances
from utils import get_train_cfg, create_hyperparameter, initial_training_config, get_train_dataset_length

from Trainer import Trainer

import os
import pickle

(
  train_dataset_name,
  train_images_path, 
  train_json_annotations_path,
  valid_dataset_name,
  valid_images_path, 
  valid_json_annotations_path,
  config_file_path, 
  checkpoint_url, 
  num_classes, 
  device, 
  cfg_save_path, 
  output_dir,
  epochs
) =  initial_training_config()

dataset_train_length = get_train_dataset_length(train_images_path)

config_hyperparameters = create_hyperparameter(
  epochs=epochs, 
  batch_size=2, 
  learning_rate=0.0001, 
  momentum=0.937, 
  weight_decay=0.0005, 
  dataset_train_length=dataset_train_length,
  early_stopping=100, 
  eval_period=50
)

register_coco_instances(
  name=train_dataset_name, 
  metadata={}, 
  json_file=train_json_annotations_path, 
  image_root=train_images_path
)

register_coco_instances(
  name=valid_dataset_name, 
  metadata={}, 
  json_file=valid_json_annotations_path, 
  image_root=valid_images_path
)

cfg = get_train_cfg(
  config_file_path, 
  checkpoint_url, 
  train_dataset_name, 
  valid_dataset_name, 
  num_classes, 
  device, 
  output_dir, 
  config_hyperparameters
)

with open (cfg_save_path, 'wb') as f:
  pickle.dump(cfg, f, protocol = pickle.HIGHEST_PROTOCOL)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = Trainer(cfg)

trainer.resume_or_load(resume=False)

trainer.train()