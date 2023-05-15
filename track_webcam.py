import cv2

from heatmap.heatmap import create_grid, draw_heatmap
from modeltracking.object_detector import YOLOv7
from utils.detections import draw


def run_yolov7_on_webcam(is_active):
    global frame
    global heatmap
    global show_frame
    global num_track

    yolov7 = YOLOv7()
    yolov7.load(r'C:\Users\LENOVO\Desktop\Test\easy-yolov7\yolov7.pt',
                classes=r'C:\Users\LENOVO\Desktop\Test\easy-yolov7\coco.yaml',
                device='gpu')  # use 'gpu' for CUDA GPU inference

    grid = create_grid()
    while is_active.value:
        detections, num_track = yolov7.detect(frame, track=True, heatmap=True if heatmap else False)
        detected_frame = draw(frame, detections)
        if heatmap:
            detected_frame = draw_heatmap(detected_frame, grid)
        show_frame = detected_frame
        # print(json.dumps(detections, indent=4))

        cv2.imshow('webcam', detected_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        else:
            break
    yolov7.unload()
