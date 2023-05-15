import cv2
import numpy as np
from skimage.transform import resize

frame_width = 640
frame_height = 480

cell_size = 40
n_cols = frame_width // cell_size
n_rows = frame_height // cell_size

heat_matrix = np.zeros((n_rows, n_cols))

def update_value(height,width):
    global frame_width, frame_height, n_cols, n_rows,heat_matrix
    frame_width = width
    frame_height = height
    n_cols = frame_width // cell_size
    n_rows = frame_height // cell_size
    heat_matrix = np.zeros((n_rows, n_cols))

def get_val():
    global frame_width, frame_height, n_cols, n_rows
    print(frame_width)
    print(frame_height)



def create_grid():
    grid = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    color = (255, 255, 255)
    thickness = 1
    for i in range(n_rows):
        start_point = (0, (i + 1) * cell_size)
        end_point = (frame_width, (i + 1) * cell_size)
        grid = cv2.line(grid, start_point, end_point, color, thickness)
    for i in range(n_cols):
        start_point = ((i + 1) * cell_size, 0)
        end_point = ((i + 1) * cell_size, frame_height)
        grid = cv2.line(grid, start_point, end_point, color, thickness)
    return grid


def reset_heatmatrix():
    global heat_matrix
    heat_matrix = np.zeros((n_rows, n_cols))


def draw_grid_on_image(image, grid):
    image = cv2.addWeighted(image, 1, grid, 0.5, 0)
    return image


def calc_heatmap(detection):
    x, y, x_plus_w, y_plus_h = detection[0], detection[1], detection[2], detection[3]
    heat_matrix[int((y_plus_h + y) // 2 // cell_size), int((x_plus_w + x) // 2 // cell_size)] += 1


def draw_heatmap(frame, grid):
    global heat_matrix
    temp_heat_matrix = heat_matrix.copy()
    temp_heat_matrix = resize(temp_heat_matrix, (frame_height, frame_width))
    temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
    temp_heat_matrix = np.uint8(temp_heat_matrix * 255)

    # Tao heat map
    image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET)
    image_heat = draw_grid_on_image(image_heat, grid)

    return draw_grid_on_image(frame, image_heat)
