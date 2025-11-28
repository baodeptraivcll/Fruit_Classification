import os
import cv2
from pathlib import Path
import random

# --------------------------
# Đường dẫn
# --------------------------
raw_folder = r"C:\Users\Asus\OneDrive\Desktop\GIT\Fruit_Classification\Dataset\raw_png"
dataset_folder = r"C:\Users\Asus\OneDrive\Desktop\GIT\Fruit_Classification\Dataset\clean"
img_size = 224

# Tạo folder train/val
(Path(dataset_folder)/"train").mkdir(parents=True, exist_ok=True)
(Path(dataset_folder)/"val").mkdir(parents=True, exist_ok=True)

# --------------------------
# Hàm đọc ảnh PNG/JPG
# --------------------------
def read_image(file_path):
    ext = str(file_path).lower()
    if not ext.endswith((".jpg", ".jpeg", ".png")):
        return None
    img = cv2.imread(str(file_path))
    return img

# --------------------------
# Xử lý dataset
# --------------------------
for class_name in os.listdir(raw_folder):
    class_input_path = Path(raw_folder) / class_name
    if not class_input_path.is_dir():
        continue

    images = list(class_input_path.glob("*.*"))
    if len(images) == 0:
        continue

    # Tạo folder train/val cho class
    (Path(dataset_folder)/"train"/class_name).mkdir(parents=True, exist_ok=True)
    (Path(dataset_folder)/"val"/class_name).mkdir(parents=True, exist_ok=True)

    # Shuffle và chia train/val 80/20
    random.shuffle(images)
    split = int(len(images)*0.8)
    train_imgs = images[:split]
    val_imgs = images[split:]

    # Đếm số thứ tự ảnh
    counter = 1

    # -------- Train -----------
    for img_path in train_imgs:
        img = read_image(img_path)
        if img is None:
            print("Skip file:", img_path)
            continue
        img = cv2.resize(img, (img_size, img_size))
        base_num = f"{counter:04d}"  # 0001, 0002, ...
        cv2.imwrite(f"{dataset_folder}/train/{class_name}/{class_name}_{base_num}.png", img)
        counter += 1

    # -------- Val -----------
    for img_path in val_imgs:
        img = read_image(img_path)
        if img is None:
            print("Skip file:", img_path)
            continue
        img = cv2.resize(img, (img_size, img_size))
        base_num = f"{counter:04d}"
        cv2.imwrite(f"{dataset_folder}/val/{class_name}/{class_name}_{base_num}.png", img)
        counter += 1

print("🎉 Dataset clean hoàn tất với tên chuẩn!")
