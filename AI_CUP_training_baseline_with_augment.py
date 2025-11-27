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
from pathlib import Path


from ultralytics import YOLO
from ultralytics.utils import metrics
import numpy as np 


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

def custom_fitness(self):
    """
    Custom fitness function that select best.pt 100% by mAP50 for the purpose of the competition
    """
    w = [0.0, 0.0, 1.0, 0.0]
    return (np.array(self.mean_results()) * w).sum()

# Save best.pt accurary to best_metrics.txt
def save_best_metrics(trainer):
    # metrics from final_eval(best.pt); keys like:
    # 'metrics/precision(B)', 'metrics/recall(B)',
    # 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    metrics = getattr(trainer, "metrics", None)
    if not metrics:
        print("No metrics found on trainer; skipping best_metrics save.")
        return

    save_path = Path(trainer.save_dir) / "best_metrics.txt"
    with open(save_path, "w") as f:
        f.write("BEST.PT FINAL METRICS\n")
        for k in ["metrics/precision(B)",
                  "metrics/recall(B)",
                  "metrics/mAP50(B)",
                  "metrics/mAP50-95(B)"]:
            if k in metrics:
                f.write(f"{k}: {metrics[k]:.4f}\n")

    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Saved best.pt metrics to {save_path}")

metrics.Metric.fitness = custom_fitness
print("Use mAP50 ")

#模型參數參考網址:https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
# to access model trained call YOLO('best.pt')
model = YOLO('yolo12n') #初次訓練使用YOLO官方的預訓練模型，如要使用自己的模型訓練可以將'yolo12n.pt'替換掉
model.add_callback("on_train_end", save_best_metrics)
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<best metric one line up >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

results = model.train(data="./aortic_valve_colab.yaml",
            epochs=80, #跑幾個epoch
            batch=16, #batch_size
            imgsz=640, #圖片大小640*640
            device=0, #使用GPU進行訓練
            degrees=5,
            patience= 20# Small rotation (±5°)
            flipud=0.0,          # Vertical flip is BAD for hearts
            # fliplr=0.5,          # Horizontal flip is OK
            scale=0.05, 
            )



