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

    def load(self, weights_path, classes, device='cpu'):
        with torch.no_grad():
            self.device = select_device(device)
            self.model = attempt_load(weights_path, device=self.device)

            if device != 'cpu':
                self.model.half()
                self.model.to(self.device).eval()

            stride = int(self.model.stride.max())
            self.imgsz = check_img_size(self.settings['img_size'], s=stride)
            self.classes = yaml.load(open(classes), Loader=yaml.SafeLoader)['classes']

    def unload(self):
        if self.device.type != 'cpu':
            torch.cuda.empty_cache()

        def __parse_image(self, img):
            im0 = img
            img = letterbox(im0, self.imgsz, auto=self.imgsz != 1280)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.device.type != 'cpu' else img.float()
            img /= 255.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            return im0, img

        def detect(self, img, track=False, heatmap=False):
            num_track = 0
            with torch.no_grad():
                im0, img = self.__parse_image(img)
                pred = self.model(img)[0]
                pred = non_max_suppression(pred, self.settings['conf_thres'], self.settings['iou_thres'])
                raw_detection = np.empty((0, 6), float)

                for det in pred:
                    if len(det) > 0:
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            raw_detection = np.concatenate((raw_detection, [
                                [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), round(float(conf), 2),
                                 int(cls)]]))

                if track:
                    raw_detection = self.tracker.update(raw_detection)
                    num_track = self.tracker.numTrackedObjects()
                    threading.Thread(target=save_detections, args=(num_track,)).start()
                if heatmap:
                    for detection in raw_detection:
                        calc_heatmap(detection)

                detections = Detections(raw_detection, self.classes, tracking=track).to_dict()
                return detections, num_track

def save_detections(num_track):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    print("vào")
    # Ghi các giá trị của các khung và thời gian vào file
    with open('detections.txt', 'a') as f:
        f.write(f'Time: {current_time}, Num people: {num_track}\n')
