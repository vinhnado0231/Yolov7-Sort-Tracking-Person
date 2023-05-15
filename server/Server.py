import asyncio
import base64
import datetime
import json
import threading

import cv2
import numpy as np
import websockets
def run_yolov7_on_webcam():
    global frame
    global heatmap
    global show_frame
    global num_track
    global is_active

    yolov7 = YOLOv7()
    yolov7.load('../yolov7.pt', classes='../coco.yaml', device='gpu')  # use 'gpu' for CUDA GPU inference

    while True:
        active = is_active
        if active:
            grid = create_grid()
            get_val()
        while active:
            detections, num_track = yolov7.detect(frame, track=True, heatmap=True if heatmap else False)
            detected_frame = draw(frame, detections)
            if heatmap:
                detected_frame = draw_heatmap(detected_frame, grid)
            show_frame = detected_frame
            # print(json.dumps(detections, indent=4))

            cv2.imshow('webcam', detected_frame)
            cv2.waitKey(1)
    yolov7.unload()


t1 = threading.Thread(target=run_yolov7_on_webcam)
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

async def server(websocket, path):
    try:
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
