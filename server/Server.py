import asyncio
import base64
import json
import threading
import time

import cv2
import numpy as np
import websockets

from firebase.firebase import get_heatmap
from heatmap.heatmap import create_grid, draw_heatmap, update_value, get_val
from modeltracking.object_detector import YOLOv7
from utils.detections import draw

yolov7 = YOLOv7()
yolov7.load('../best_416.pt', classes='../mydataset.yaml', device='gpu')  # use 'gpu' for CUDA GPU inference

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
    global is_active
    global heatmap
    while True:
        active = is_active
        if active:
            time.sleep(0.5)
            grid = create_grid()
            get_val()
        while active:
            detected_frame = draw(frame, detections)
            heatmap_copy = heatmap
            if heatmap_copy:
                detected_frame = draw_heatmap(detected_frame, grid)
            show_frame = detected_frame
            cv2.imshow('webcam', detected_frame)
            cv2.waitKey(1)


def update_heatmap():
    global heatmap
    while True:
        heatmap_copy = heatmap
        heatmap = get_heatmap(heatmap_copy)
        time.sleep(1)


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
            active_heatmap = heatmap
            detections, num_track = yolov7.detect(frame, track=True, heatmap=True if active_heatmap else False)
            if check:
                check = False
                threading.Thread(target=resolve_frame).start()
                threading.Thread(target=update_heatmap).start()

    yolov7.unload()


threading.Thread(target=run_yolov7_on_server).start()


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



async def handle_get_frame(websocket):
    global show_frame
    global num_track
    while True:
        show_frame = show_frame
        num_track = num_track

        frame_dict = {
            "frame": base64.b64encode(cv2.imencode('.jpg', show_frame)[1]).decode('utf-8'),
        }

        num_dict = {"Number of people": num_track}

        data_dict = {**frame_dict, **num_dict}

        # Gửi dictionary đến client
        await websocket.send(json.dumps(data_dict))


async def path1(websocket, path):
    try:
        if path == "/send_frame":
            await handle_send_frame(websocket)
    except Exception as e:
        print(f"Error occurred: {e}")


async def path2(websocket, path):
    try:
        if path == "/get_frame":
            await handle_get_frame(websocket)
    except Exception as e:
        print(f"Error occurred: {e}")


print("Turn on server")


def websocket1():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(path1, "127.0.0.1", 9000, ping_interval=None)
    loop.run_until_complete(start_server)
    loop.run_forever()


def websocket2():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(path2, "127.0.0.1", 9001, ping_interval=None)
    loop.run_until_complete(start_server)
    loop.run_forever()

threading.Thread(target=websocket1).start()
threading.Thread(target=websocket2).start()
