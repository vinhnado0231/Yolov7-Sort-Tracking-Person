import cv2
import numpy as np

# đường dẫn tới các file YOLOv4-tiny
model_config = "yolov4-tiny.cfg"
model_weights = "yolov4-tiny.weights"

# khởi tạo mô hình YOLOv4-tiny với tệp cấu hình và trọng số được chỉ định
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# lấy tên lớp đầu ra (ở đây chỉ có một lớp "person")
layer_names = net.getLayerNames()
output_layer = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# mở camera
cap = cv2.VideoCapture(0)

while True:
    # đọc frame từ camera
    ret, frame = cap.read()

    # thực hiện xử lý đối tượng và trả về bounding box và độ tin cậy
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
            if class_id == 0 and confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # vẽ bounding box và ghi nhãn con người nhận diện được
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indices) != 0:
        for i in indices.ravel():
            x, y, w, h = boxes[i]
            label = f"Person {confidences[i]:.2f}"
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), font, 0.5, color, 2)

    # hiển thị frame với nhãn nhận diện
    cv2.imshow("Detection", frame)

    # thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# giải phóng camera
cap.release()

# đóng tất cả cửa sổ hiển thị
cv2.destroyAllWindows()