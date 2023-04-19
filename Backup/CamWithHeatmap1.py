import json

import cv2
import numpy as np
import base64
import asyncio
import websockets
import threading

# Parameters
classnames_file = "classnames.txt"
weights_file = "yolov4-tiny.weights"
config_file = "yolov4-tiny.cfg"
conf_threshold = 0.5
nms_threshold = 0.4
detect_class = "person"

# mở camera
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cell_size = 65  # 40x40 pixel
n_cols = frame_width // cell_size
n_rows = frame_height // cell_size
alpha = 0.4

heat_matrix = np.zeros((n_rows, n_cols))
# scale = 0.00392  # Chua tim ra nguon goc

yolo_net = cv2.dnn.readNet(weights_file, config_file)

layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Doc ten cac class
classes = None
with open(classnames_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


# Ham tra ve dong va cot tu toa do x, y
def get_row_col(x, y):
    row = y // cell_size
    col = x // cell_size
    return row, col


def draw_grid(image):
    for i in range(n_rows):
        start_point = (0, (i + 1) * cell_size)
        end_point = (frame_width, (i + 1) * cell_size)
        color = (255, 255, 255)
        thickness = 1
        image = cv2.line(image, start_point, end_point, color, thickness)

    for i in range(n_cols):
        start_point = ((i + 1) * cell_size, 0)
        end_point = ((i + 1) * cell_size, frame_height)
        color = (255, 255, 255)
        thickness = 1
        image = cv2.line(image, start_point, end_point, color, thickness)

    return image


# Ham ve cac hinh chu nhat va ten class
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    global heat_matrix
    r, c = get_row_col((x_plus_w + x) // 2, (y_plus_h + y) // 2)
    heat_matrix[r, c] += 1

    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

frame = 0

def calc():
    global frame
    while True:
        ret, frame = cap.read()

        # thực hiện xử lý đối tượng và trả về bounding box và độ tin cậy
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        layer_outputs = yolo_net.forward(output_layers)
        boxes = []
        confidences = []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > conf_threshold:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        # vẽ bounding box và ghi nhãn con người nhận diện được
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Ve cac khung chu nhat quanh doi tuong
        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(frame, 0, round(x), round(y), round(x + w), round(y + h))

        cv2.putText(frame, f"Number of people: {len(indices)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        from skimage.transform import resize

        temp_heat_matrix = heat_matrix.copy()
        temp_heat_matrix = resize(temp_heat_matrix, (frame_height, frame_width))
        temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
        temp_heat_matrix = np.uint8(temp_heat_matrix * 255)

        # Tao heat map
        image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET)

        frame = draw_grid(frame)

        # Chong hinh
        cv2.addWeighted(image_heat, alpha, frame, 1 - alpha, 0, frame)


        # cv2.imshow("Detection", frame)
        # thoát nếu nhấn phím 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # giải phóng camera
    cap.release()

    # đóng tất cả cửa sổ hiển thị
    cv2.destroyAllWindows()


t1 = threading.Thread(target=calc)
t1.start()

async def video_feed(websocket):
    while cap.isOpened():
        # Capture the current frame
        ret, frame1 = cap.read()
        if not ret:
            break

        # Capture the next frame
        ret, frame2 = cap.read()
        if not ret:
            break

        # Encode the frames as JPEG images and add them to the dictionary
        frame_dict = {
            "frame1": base64.b64encode(cv2.imencode('.jpg', frame1)[1]).decode('utf-8'),
            "frame2": base64.b64encode(cv2.imencode('.jpg', frame2)[1]).decode('utf-8')
        }

        # Create a random matrix and add it to the dictionary
        matrix = np.random.rand(3, 3)
        matrix_dict = {"matrix": base64.b64encode(matrix.tobytes()).decode('utf-8')}

        # Combine the two dictionaries
        data_dict = {**frame_dict, **matrix_dict}

        # Send the dictionary to the client
        await websocket.send(json.dumps(data_dict))

    cap.release()


print("Turn on server")
start_server = websockets.serve(video_feed, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()