import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding


# !pip install ultralytics
import ultralytics
ultralytics.checks()

#下載資料集
import os
import shutil

# Upload training_image.zip, training_label.zip, aortic_valve_colab.yaml and put it in /content/


#移動檔案
def find_patient_root(root):
    """往下找，直到找到含有 patientXXXX 的資料夾"""
    for dirpath, dirnames, filenames in os.walk(root):
        if any(d.startswith("patient") for d in dirnames):
            return dirpath
    return root  # fallback

# 解壓縮到固定資料夾
if not os.path.isdir("./training_image") and os.path.exists("training_image.zip"):
    os.makedirs("./training_image", exist_ok=True)

if not os.path.isdir("./training_label") and os.path.exists("training_label.zip"):
    os.makedirs("./training_label", exist_ok=True)

IMG_ROOT = find_patient_root("./training_image")
LBL_ROOT = find_patient_root("./training_label")

print("IMG_ROOT =", IMG_ROOT)
print("LBL_ROOT =", LBL_ROOT)

# 建立並清空輸出資料夾（若存在）
def ensure_clean_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

ensure_clean_dir("./datasets/train/images")
ensure_clean_dir("./datasets/train/labels")
ensure_clean_dir("./datasets/val/images")
ensure_clean_dir("./datasets/val/labels")

def move_patients(start, end, split):
    for i in range(start, end + 1):
        patient = f"patient{i:04d}"
        img_dir = os.path.join(IMG_ROOT, patient)
        lbl_dir = os.path.join(LBL_ROOT, patient)
        if not os.path.isdir(lbl_dir):
            continue

        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue

            label_path = os.path.join(lbl_dir, fname)
            base, _ = os.path.splitext(fname)  # 取出檔名不含副檔名
            img_path = os.path.join(img_dir, base + ".png")
            if not os.path.exists(img_path):
                print(f"找不到對應圖片: {img_path}")
                continue

            shutil.copy2(img_path, f"./datasets/{split}/images/")
            shutil.copy2(label_path, f"./datasets/{split}/labels/")

# patient0001~0030 → train
move_patients(1, 30, "train")

# patient0031~0050 → val
move_patients(31, 50, "val")

print("完成移動！")


from ultralytics import YOLO
#模型參數參考網址:https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
# to access model trained call YOLO('best.pt')
model = YOLO('yolov12n.pt') #初次訓練使用YOLO官方的預訓練模型，如要使用自己的模型訓練可以將'yolo12n.pt'替換掉
results = model.train(
    data="./aortic_valve_colab.yaml",
    epochs=110,
    batch=16,
    imgsz=640,
    device=0,
    patience = 

    # --- CT-friendly augmentations ---
    degrees=10,          # small rotation
    translate=0.05,      # slight shift
    scale=0.10,          # slight zoom
    shear=0.0,           # do NOT shear medical images
    perspective=0.0,     # never change CT geometry

    flipud=0.0,          # DO NOT flip vertically (anatomical distortion)
    fliplr=0.10,         # 10% horizontal flip is OK

    hsv_h=0.0,           # CT images are grayscale
    hsv_s=0.0,
    hsv_v=0.0,

    # brightness=0.10,     # mild brightness change
    # contrast=0.10,       # mild contrast change
    # gaussian=0.01,       # CT noise simulation

    mosaic=0.0,          # NEVER use mosaic on medical CT
    mixup=0.0,           # NEVER mixup anatomy across patients

    copy_paste=0.0,      # unrealistic for organs
)
