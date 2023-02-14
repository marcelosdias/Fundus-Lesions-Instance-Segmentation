import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_train_loader

import os

from LossEvalHook import LossEvalHook

from utils import *

import torch
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import CfgNode
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.engine import DefaultTrainer
import torch

import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_train_loader

from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)

import os

from detectron2.config import CfgNode

from CocoEvaluator import CocoEvaluator

cfg_save_path = 'config.pickle'
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
        return CocoEvaluator(dataset_name, cfg, True, output_folder)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            5000,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks
    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
                weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )
        return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
            cfg, mapper= DatasetMapper(cfg, is_train=True, augmentations=[
                T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'),
                T.RandomFlip(),
                T.RandomBrightness(0.8, 1.2),
                T.RandomContrast(0.8, 1.2),
                T.RandomSaturation(0.8, 1.2),
                T.RandomLighting(0.7),
            ])
        )