import threading
import warnings
import datetime

from heatmap.heatmap import calc_heatmap
from sort.sort import Sort

warnings.filterwarnings('ignore')
from utils.general import check_img_size, non_max_suppression, scale_coords
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.detections import Detections
from utils.datasets import letterbox
import numpy as np
import torch
import yaml


class YOLOv7:
    def __init__(self, conf_thres=0.5, iou_thres=0.45, img_size=640):
        self.settings = {
            'conf_thres': conf_thres,
            'iou_thres': iou_thres,
            'img_size': img_size,
        }
        self.tracker = Sort()

