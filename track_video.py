import cv2
from tqdm import tqdm

from heatmap.heatmap import draw_heatmap, create_grid
from modeltracking.object_detector import YOLOv7
from utils.detections import draw

yolov7 = YOLOv7()
yolov7.load('yolov7.pt', classes='coco.yaml', device='gpu')  # use 'gpu' for CUDA GPU inference

video = cv2.VideoCapture('video.mp4')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

if video.isOpened() == False:
    print('[!] error opening the video')

print('[+] tracking video...\n')
pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)
heatmap = True
grid = create_grid()
try:
    while video.isOpened():
        ret, frame = video.read()
        if ret == True:
            detections, num_track = yolov7.detect(frame, track=True)
            detected_frame = draw(frame, detections)
            output.write(detected_frame)
            pbar.update(1)
        else:
            break
except KeyboardInterrupt:
    pass

pbar.close()
video.release()
output.release()
yolov7.unload()
