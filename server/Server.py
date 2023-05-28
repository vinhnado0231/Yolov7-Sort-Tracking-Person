import asyncio
import base64
import datetime
import json
import threading

import cv2
import firebase_admin
import numpy as np
import websockets
from firebase_admin import credentials, db

from heatmap.heatmap import create_grid, draw_heatmap, update_value, get_val
from modeltracking.object_detector import YOLOv7
from utils.detections import draw

frame = 0
is_active = False
show_frame = 0
num_track = 0
heatmap = True

import cv2

def show_camera():
    global frame
    # Mở camera
    cap = cv2.VideoCapture(0)  # Số 0 cho biết sử dụng camera mặc định

    while True:
        # Đọc khung hình từ camera
        ret, frame1 = cap.read()
        frame=frame1

        # Kiểm tra phím bấm ESC để thoát
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()



index_frame = 0

def save_frame():
    global index_frame
    global show_frame
    show_frame = show_frame
    index_frame = index_frame + 1
    new_frame = base64.b64encode(cv2.imencode('.jpg', show_frame)[1]).decode('utf-8')
    ref_frame = db.reference(f"/frame")
    ref_frame.set(new_frame)



def run_yolov7_on_server():
    global frame
    global heatmap
    global show_frame
    global num_track
    global is_active

    yolov7 = YOLOv7()
    yolov7.load('../yolov7.pt', classes='../coco.yaml', device='cpu')  # use 'gpu' for CUDA GPU inference

    while True:
        active = is_active
        if True:
            grid = create_grid()
            get_val()
        while True:
            detections, num_track = yolov7.detect(frame, track=True, heatmap=True if heatmap else False)
            detected_frame = draw(frame, detections)
            if heatmap:
                detected_frame = draw_heatmap(detected_frame, grid)
            show_frame = detected_frame
            threading.Thread(target=save_frame).start()

            # print(json.dumps(detections, indent=4))

            cv2.imshow('webcam', detected_frame)
            cv2.waitKey(1)
    yolov7.unload()


t2 = threading.Thread(target=show_camera)
t2.start()
cv2.waitKey(3000)
t1 = threading.Thread(target=run_yolov7_on_server)
t1.start()


async def handle_send_frame(websocket):
    global frame
    global is_active
    update = False
    while True:
        # Receive a frame from the server
        data = await websocket.recv()

        # Convert the JPEG-encoded image to a NumPy array
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if not update:
            update = True
            height, width = frame.shape[:2]
            update_value(height, width)
        is_active = True
        # cv2.imshow("Detection", frame)
        cv2.waitKey(1)


async def handle_get_frame(websocket):
    global show_frame
    global num_track
    while True:
        show_frame = show_frame
        num_track = num_track
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]

        time_dict = {"Time": current_time}

        frame_dict = {
            "frame": base64.b64encode(cv2.imencode('.jpg', show_frame)[1]).decode('utf-8'),
        }

        num_dict = {"Number of people": num_track}

        data_dict = {**frame_dict, **num_dict, **time_dict}

        # Gửi dictionary đến client
        await websocket.send(json.dumps(data_dict))


async def server(websocket, path):
    try:
        print("request" + path)
        if path == "/get_frame":
            await handle_get_frame(websocket)
        elif path == "/get_data":
            with open("detections.txt", "r") as f:
                content = f.read()
            await websocket.send(content)
        elif path == "/send_frame":
            await handle_send_frame(websocket)
    except Exception as e:
        print(f"Error occurred: {e}")


print("Turn on server")
start_server = websockets.serve(server, "localhost", 8765, ping_interval=None)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
