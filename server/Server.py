import asyncio
import base64
import datetime
import json
import threading
import time

import cv2
import numpy as np
import websockets

from heatmap.heatmap import create_grid, draw_heatmap, update_value, get_val
from modeltracking.object_detector import YOLOv7
from utils.detections import draw

yolov7 = YOLOv7()
yolov7.load('../best.pt', classes='../mydataset.yaml', device='gpu')  # use 'gpu' for CUDA GPU inference

frame = 0
detections = 0
is_active = False
show_frame = 0
num_track = 0
heatmap = True

def resolve_frame():
    global frame
    global detections
    global show_frame
    global  is_active
    while True:
        active = is_active
        if active:
            time.sleep(1)
            grid = create_grid()
            get_val()
        while active:
            detected_frame = draw(frame, detections)
            if heatmap:
                detected_frame = draw_heatmap(detected_frame, grid)
            show_frame = detected_frame
            cv2.imshow('webcam', detected_frame)
            cv2.waitKey(1)

def run_yolov7_on_server():
    global frame
    global heatmap
    global num_track
    global is_active
    global detections

    while True:
        active = is_active
        check = True
        while active:
            detections, num_track = yolov7.detect(frame, track=True, heatmap=True if heatmap else False)
            if check:
                check = False
                t2 = threading.Thread(target=resolve_frame)
                t2.start()

    yolov7.unload()


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

async def handle_control(websocket):
    check = True
    while True:

        if check:
            check=False
            await websocket.send("Turn left")

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
        if path == "/get_frame":
            await handle_get_frame(websocket)
        elif path == "/control":
            await handle_control(websocket)
        elif path == "/send_frame":
            await handle_send_frame(websocket)
    except Exception as e:
        print(f"Error occurred: {e}")


print("Turn on server")
start_server = websockets.serve(server, "127.0.0.1", 9000, ping_interval=None)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
