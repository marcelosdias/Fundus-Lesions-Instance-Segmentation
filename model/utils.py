from detectron2.utils.logger import setup_logger
import argparse

setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg

import os

def create_hyperparameter(epochs, batch_size, learning_rate, momentum, weight_decay, dataset_train_length, early_stopping, eval_period):
    MAX_ITER = convert_epochs_to_max_iter(epochs, dataset_train_length, batch_size)
    PATIENCE = convert_epochs_to_max_iter(early_stopping, dataset_train_length, batch_size)
    EVAL_PERIOD = convert_epochs_to_max_iter(eval_period, dataset_train_length, batch_size)

    return {
        'EPOCHS': epochs,
        'IMS_PER_BATCH': batch_size,
        'BASE_LR': learning_rate,
        'MOMENTUM': momentum,
        'WEIGHT_DECAY': weight_decay,
        'MAX_ITER': MAX_ITER,
        'EVAL_PERIOD': EVAL_PERIOD,
        'PATIENCE': PATIENCE
    }

def convert_epochs_to_max_iter(epochs, num_images, ims_per_batch, num_gpus = 1):
    single_iteration =  num_gpus * ims_per_batch

    iterations_for_one_epoch = num_images / single_iteration

    max_iter = iterations_for_one_epoch * epochs

    return int(max_iter)

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, valid_dataset_name, num_classes, device, output_dir, config_hyperparameters):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)

    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (valid_dataset_name,)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5

    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = config_hyperparameters['IMS_PER_BATCH']
    cfg.SOLVER.MAX_ITER = config_hyperparameters['MAX_ITER']
    cfg.SOLVER.BASE_LR = config_hyperparameters['BASE_LR']
    cfg.SOLVER.MOMENTUM = config_hyperparameters['MOMENTUM']
    cfg.SOLVER.WEIGHT_DECAY = config_hyperparameters['WEIGHT_DECAY']

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    cfg.TEST.EVAL_PERIOD = config_hyperparameters['EVAL_PERIOD']

    cfg.SOLVER.STEPS = []

    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    cfg.PATIENCE = config_hyperparameters['PATIENCE']

    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 18000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 9000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1500

    cfg.TEST.DETECTIONS_PER_IMAGE = 256

    return cfg

def initial_training_config():
    config_file_path = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
    checkpoint_url = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'

    num_classes = 4
    device = 'cuda'

    cfg_save_path = 'config.pickle'

    output_dir = './output'

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset')

    parser.add_argument('--epochs')

    args = parser.parse_args()

    train_dataset_name, train_images_path, train_json_annotations_path = get_dataset_config(args.dataset, 'train')

    valid_dataset_name, valid_images_path, valid_json_annotations_path = get_dataset_config(args.dataset, 'valid')

    return (
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
        int(args.epochs)
    )

def get_dataset_config(dataset, type):
    if dataset == 'ddr':
        return (
            f'{type}_dataset',
            f'../datasets/datasets-cropping-tilling/ddr/{type}',
            f'../datasets/datasets-cropping-tilling/ddr/{type}/_annotations.coco.json',
        )
    elif 'idrid':
        return (
            f'{type}_dataset',
            f'../datasets/datasets-cropping-tilling/idrid/{type}',
            f'../datasets/datasets-cropping-tilling/idrid/{type}/_annotations.coco.json',
        )
    else:
        raise Exception("Invalid dataset")

def get_train_dataset_length(dataset_path):
    length = 0

    for path in os.listdir(dataset_path):
        if os.path.isfile(os.path.join(dataset_path, path)):
            length += 1

    return length - 1