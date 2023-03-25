import cv2
import numpy as np
from sort import Sort

# đường dẫn tới các file YOLOv4-tiny
model_config = "yolov4-tiny.cfg"
model_weights = "yolov4-tiny.weights"

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

layer_names = net.getLayerNames()
output_layer = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

thres = 0.5
cap = cv2.VideoCapture(0)

# tạo một mảng chứa các thông tin về vật thể phát hiện được
detections = []

while True:
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer)
    boxes = []
    confidences = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > thres:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    count = 0

    for i in indices.ravel():
        x, y, w, h = boxes[i]
        label = f"Person {confidences[i]:.2f}"
        detection = [x, y, x + w, y + h, confidences[i]]
        detections = np.vstack((detections, detection))

    # sử dụng thuật toán SORT để theo dõi các vật thể
    if len(detections) > 0:
        detections = np.array(detections)
        tracker = Sort()
        tracked_objects = tracker.update(detections)

        # hiển thị các vật thể được theo dõi
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(np.int32)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {track_id}", (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Number of people: {count}", (10, 30), font, 0.8, (0, 0, 255), 2)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()