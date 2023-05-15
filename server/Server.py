import asyncio
import base64
import datetime
import json
import threading

import cv2
import numpy as np
import websockets


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
