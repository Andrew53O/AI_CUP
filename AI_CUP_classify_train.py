import os
import shutil

TRAINIMG_ROOT = "./training_image/"
TRAINLBL_ROOT = "./training_label/"
POSITIVE_ROOT = "./datasets_cls/train/positive/"
NEGATIVE_ROOT = "./datasets_cls/train/negative/"
POSITIVE_VAL_ROOT = "./datasets_cls/val/positive/"
NEGATIVE_VAL_ROOT = "./datasets_cls/val/negative/"

def move_patients(start, end, split):
    for i in range(start, end+1):
        print("i=",i)
        patient = f"patient{i:04d}"
        img_dir = os.path.join(TRAINIMG_ROOT, patient)
        lbl_dir = os.path.join(TRAINLBL_ROOT, patient)

        for fname in os.listdir(img_dir):
            if not fname.endswith(".png"):
                continue
            base, _ = os.path.splitext(fname)  # 取出檔名不含副檔名
            label_path = os.path.join(lbl_dir, base + ".txt")
            img_path = os.path.join(img_dir, fname)

            if os.path.exists(label_path):
                if split == "val":
                    shutil.copy2(img_path, POSITIVE_VAL_ROOT)
                else:
                    shutil.copy2(img_path, POSITIVE_ROOT)
            else:
                if split == "val":
                    shutil.copy2(img_path, NEGATIVE_VAL_ROOT)
                else:
                    shutil.copy2(img_path, NEGATIVE_ROOT)

os.makedirs(POSITIVE_ROOT, exist_ok=True)
os.makedirs(NEGATIVE_ROOT, exist_ok=True)
os.makedirs(POSITIVE_VAL_ROOT, exist_ok=True)
os.makedirs(NEGATIVE_VAL_ROOT, exist_ok=True)
# move_patients(1, 40, "train")
# move_patients(41, 50, "val")

from ultralytics import YOLO
import os
print("CWD =", os.getcwd())
model = YOLO("yolo11n-cls.pt")
model.train(
    data="/home/undergraduate/AI_CUP/datasets_cls",
    epochs=80, #跑幾個epoch
    batch=16, #batch_size
    imgsz=640, #圖片大小640*640
    device=0, #
    patience=30
)