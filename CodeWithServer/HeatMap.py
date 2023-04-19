import time

import cv2
import numpy as np
from skimage.transform import resize

def draw_heatmap_from_boxes(boxes, frame_height, frame_width, cell_size, alpha):
    # Khởi tạo ma trận heat với kích thước bằng với số ô trên mỗi chiều
    n_rows = frame_height // cell_size
    n_cols = frame_width // cell_size
    heat_matrix = np.zeros((n_rows, n_cols))

    # Cập nhật ma trận heat dựa trên các khung đã detect
    for box in boxes:
        x, y, w, h = box
        # Tính toán vị trí của ô trên ma trận heat tương ứng với khung
        col_start = max(int(x / cell_size), 0)
        row_start = max(int(y / cell_size), 0)
        col_end = min(int((x + w) / cell_size) + 1, n_cols)
        row_end = min(int((y + h) / cell_size) + 1, n_rows)
        # Cập nhật giá trị của các ô
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                heat_matrix[i][j] += 1

    # Chuẩn hóa giá trị trong ma trận heat và tạo ra ảnh heatmap
    temp_heat_matrix = resize(heat_matrix, (frame_height, frame_width))
    temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
    temp_heat_matrix = np.uint8(temp_heat_matrix * 255)
    image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET)

    # Vẽ lưới trên ảnh và áp dụng chống hình
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    for i in range(n_rows):
        start_point = (0, (i + 1) * cell_size)
        end_point = (frame_width, (i + 1) * cell_size)
        color = (255, 255, 255)
        thickness = 1
        frame = cv2.line(frame, start_point, end_point, color, thickness)
    for i in range(n_cols):
        start_point = ((i + 1) * cell_size, 0)
        end_point = ((i + 1) * cell_size, frame_height)
        color = (255, 255, 255)
        thickness = 1
        frame = cv2.line(frame, start_point, end_point, color, thickness)
    cv2.addWeighted(image_heat, alpha, frame, 1 - alpha, 0, frame)

    return frame
