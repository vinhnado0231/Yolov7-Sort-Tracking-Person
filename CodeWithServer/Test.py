import cv2
import base64
import asyncio
import websockets

async def video_feed(websocket):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        print("ok")
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame from camera")
            break

        # Convert the frame to base64 and send it to the client
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        await websocket.send(jpg_as_text)

    cap.release()

start_server = websockets.serve(video_feed, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()