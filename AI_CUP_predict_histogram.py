import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# !pip install ultralytics

import ultralytics
ultralytics.checks()

import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Histogram Equalization for CT
# -----------------------------
def equalize_folder(folder, method="clahe"):
    """
    對指定資料夾內所有 .png 影像做灰階直方圖均衡化，
    然後覆寫原檔（存成 3-channel 圖片，方便 YOLO 使用）。
    """
    if not os.path.isdir(folder):
        print(f"[WARN] 資料夾不存在：{folder}")
        return

    cnt = 0
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".png"):
            continue

        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] 無法讀取圖片: {path}")
            continue

        if method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            eq = clahe.apply(img)
        else:
            eq = cv2.equalizeHist(img)

        # 灰階 → 3-channel，給 YOLO 用
        eq_rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(path, eq_rgb)
        cnt += 1

    print(f"{folder} 已完成直方圖均衡化，共處理 {cnt} 張圖片（method={method}）。")


# ==============================
# 1. 將 testing 影像從 tmp 分成 images1 / images2
# ==============================

# 要把 testing 的資料上傳到 server (testing.zip)，解壓縮到 ./datasets/test/tmp
base_root = "./datasets/test/tmp"
dst_root1 = "./datasets/test/images1"
dst_root2 = "./datasets/test/images2"

# 確保目標資料夾存在
os.makedirs(dst_root1, exist_ok=True)
os.makedirs(dst_root2, exist_ok=True)

# 自動找到第一個「直屬子資料夾含 patient*」的目錄
patient_root = base_root
for dirpath, dirnames, _ in os.walk(base_root):
    if any(d.lower().startswith("patient") for d in dirnames):
        patient_root = dirpath
        break

# 收集所有圖片路徑（只看直屬的 patient 資料夾）
all_files = []
for patient_folder in os.listdir(patient_root):
    patient_path = os.path.join(patient_root, patient_folder)
    if os.path.isdir(patient_path) and patient_folder.lower().startswith("patient"):
        for fname in os.listdir(patient_path):
            if fname.lower().endswith(".png"):
                all_files.append(os.path.join(patient_path, fname))

# 按名稱排序並對半移動
all_files.sort()
half = len(all_files) // 2

for f in all_files[:half]:
    shutil.move(f, os.path.join(dst_root1, os.path.basename(f)))

for f in all_files[half:]:
    shutil.move(f, os.path.join(dst_root2, os.path.basename(f)))

print(f"來源根目錄：{patient_root}")
print(f"完成移動！總共 {len(all_files)} 張，前半 {half} 張到 images1，後半 {len(all_files)-half} 張到 images2")

print('測試集圖片數量 : ',
      len(os.listdir("./datasets/test/images1")) +
      len(os.listdir("./datasets/test/images2")))

# ==============================
# 2. 對 images1 / images2 做 Histogram Equalization
# ==============================
equalize_folder(dst_root1, method="clahe")  # 和訓練時一樣用 CLAHE
equalize_folder(dst_root2, method="clahe")

# ==============================
# 3. 預測第一半 (images1)
# ==============================

# 這邊下載主辦方給的影像辨識模型(best.pt)，執行前要先放到 ./best.pt
# 模型參數參考網址:
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
model = YOLO('./best.pt')

results = model.predict(
    source="./datasets/test/images1/",
    save=True,
    imgsz=640,
    device=0
)

# 如果你不是在 Colab，用不到 Image 也沒關係，留著也不影響
try:
    from IPython.display import Image
    # 把預測的其中一張圖畫出來（依照你的習慣）
    # 注意：檔名要換成實際存在的那一張
    Image(filename='./runs/detect/predict/patient0051_0260.jpg', height=600)
except ImportError:
    pass

# 預測數量
print("images1 預測張數:", len(results))

# 範例：取得某一張圖的預測資訊（index 要在範圍內）
if len(results) > 260 and len(results[260].boxes) > 0:
    print('預測類別 : ', results[260].boxes.cls[0].item())
    print('預測信心分數 : ', results[260].boxes.conf[0].item())
    print('預測框座標 : ', results[260].boxes.xyxy[0].tolist())

# 將偵測框數值寫進.txt檔
os.makedirs('./predict_txt', exist_ok=True)
output_file = open('./predict_txt/images1.txt', 'w', encoding="utf-8")
for i in range(len(results)):
    # 取得圖片檔名（不含副檔名）
    filename = results[i].path.split('/')[-1].split('.png')[0]

    # 取得預測框數量
    boxes = results[i].boxes
    box_num = len(boxes.cls.tolist())

    # 如果有預測框
    if box_num > 0:
        for j in range(box_num):
            # 提取資訊
            label = int(boxes.cls[j].item())  # 類別
            conf = boxes.conf[j].item()       # 信心度
            x1, y1, x2, y2 = boxes.xyxy[j].tolist()  # 邊界框座標

            # 建立一行資料
            line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
            output_file.write(line)

# 關閉輸出檔案
output_file.close()

# 釋放記憶體
import torch
del boxes, all_files, results
torch.cuda.empty_cache()

# ==============================
# 4. 預測第二半 (images2)
# ==============================
model = YOLO('./best45.pt')

results = model.predict(
    source="./datasets/test/images2/",
    save=True,
    imgsz=640,
    device=0
)

output_file = open('./predict_txt/images2.txt', 'w', encoding="utf-8")
for i in range(len(results)):
    # 取得圖片檔名（不含副檔名）
    filename = results[i].path.split('/')[-1].split('.png')[0]

    # 取得預測框數量
    boxes = results[i].boxes
    box_num = len(boxes.cls.tolist())

    # 如果有預測框
    if box_num > 0:
        for j in range(box_num):
            # 提取資訊
            label = int(boxes.cls[j].item())  # 類別
            conf = boxes.conf[j].item()       # 信心度
            x1, y1, x2, y2 = boxes.xyxy[j].tolist()  # 邊界框座標

            # 建立一行資料
            line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
            output_file.write(line)

# 關閉輸出檔案
output_file.close()

# ==============================
# 5. 合併兩個 txt
# ==============================
file1 = "./predict_txt/images1.txt"
file2 = "./predict_txt/images2.txt"
output = "./predict_txt/merged.txt"

with open(output, "w", encoding="utf-8") as fout:
    for f in [file1, file2]:
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8") as fin:
                fout.writelines(fin.readlines())

print(f"合併完成 -> {output}")
