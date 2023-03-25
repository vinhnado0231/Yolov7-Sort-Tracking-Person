import os

# Đường dẫn tới folder chứa các file ảnh và label
folder_path = r"C:\Users\vinhn\OneDrive\Máy tính\Train\yolotinyv4_human-detect_demo\obj"

# Lấy danh sách các file trong folder
files = os.listdir(folder_path)

# Tạo danh sách các file ảnh và label tương ứng
image_files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]
label_files = [f for f in files if f.endswith(".txt")]

# Xoá các file không có tương ứng
for f in image_files:
    label_file = os.path.splitext(f)[0] + ".txt"
    if label_file not in label_files:
        os.remove(os.path.join(folder_path, f))

for f in label_files:
    image_file = os.path.splitext(f)[0] + ".jpg"
    if image_file not in image_files:
        os.remove(os.path.join(folder_path, f))
