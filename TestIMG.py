import cv2
import numpy as np

# đường dẫn tới các file YOLOv4-tiny
model_config = "yolov4-tiny.cfg"
model_weights = "yolov4-tiny.weights"

# đường dẫn tới hình ảnh đầu vào và đầu ra
input_image_path = "input.jpg"
output_image_path = "output.jpg"

# đọc hình ảnh đầu vào
img = cv2.imread(input_image_path)

# khởi tạo mô hình YOLOv4-tiny với tệp cấu hình và trọng số được chỉ định
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# lấy tên lớp đầu ra (ở đây chỉ có một lớp "person")
layer_names = net.getLayerNames()
output_layer = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# thực hiện xử lý đối tượng và trả về bounding box và độ tin cậy
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
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
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            w = int(detection[2] * img.shape[1])
            h = int(detection[3] * img.shape[0])
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))

# vẽ bounding box và ghi nhãn con người nhận diện được
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_SIMPLEX
colors = np.random.uniform(0, 255, size=(len(boxes), 3))
for i in indices.ravel():
    x, y, w, h = boxes[i]
    label = f"Person {confidences[i]:.2f}"
    color = colors[i]
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, label, (x, y-10), font, 0.5, color, 2)

# ghi hình ảnh nhận diện ra file đầu ra
cv2.imwrite(output_image_path, img)

