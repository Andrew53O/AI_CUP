import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# !pip install ultralytics

import ultralytics
ultralytics.checks()

import os

# !mkdir ./datasets
# !mkdir ./datasets/test

# 要把testing的資料上傳到server (testing.zip)，在解壓縮到/datasets/test/tmp


################# MODEL SETTING ##########################

MODEL_USED = './best49-1.pt'

################# MODEL SETTING #######################


import os
import shutil

base_root = "./datasets/test/tmp"
dst_root1 = "./datasets/test/images1"
dst_root2 = "./datasets/test/images2"


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

print('測試集圖片數量 : ',len(os.listdir("./datasets/test/images1"))+len(os.listdir("./datasets/test/images2")))

#####################################################################
# 這邊下載主辦方給的影像辨識模型(best.pt)，執行前要先放到/content/best.pt
#####################################################################

from ultralytics import YOLO
#模型參數參考網址:https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
model = YOLO(MODEL_USED)
results = model.predict(source="./datasets/test/images1/",
              save=True,
              imgsz=640,   
              device=0
              )

from IPython.display import Image

# 把預測的其中一張圖畫出來
Image(filename='./runs/detect/predict/patient0051_0260.jpg', height=600)


# 預測數量
print(len(results))

# 取得預測結果的資訊
# 想取得更多資訊可參考官方說明
# https://docs.ultralytics.com/zh/modes/predict/#boxes
print('預測類別 : ',results[260].boxes.cls[0].item())
print('預測信心分數 : ',results[260].boxes.conf[0].item())
print('預測框座標 : ',results[260].boxes.xyxy[0].tolist())


# 將偵測框數值寫進.txt檔

##################################
# !mkdir ./predict_txt/
# 先建立一個資料夾是./predict_txt/
##################################

output_file = open('./predict_txt/images1.txt', 'w')
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
# 因為Colab RAM不會自動釋放，所以透過程式釋放
import torch

# 刪除大型變數
del boxes,all_files,results
torch.cuda.empty_cache()


# 預測後半圖片(執行時間3分鐘)
from ultralytics import YOLO
#模型參數參考網址:https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
model = YOLO(MODEL_USED)
results = model.predict(source="./datasets/test/images2/",
              save=True,
              imgsz=640,
              device=0
              )


# 將後半部分偵測框數值寫進.txt檔
output_file = open('./predict_txt/images2.txt', 'w')
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


# 合併兩個.txt
file1 = "./predict_txt/images1.txt"
file2 = "./predict_txt/images2.txt"
output = "./predict_txt/merged.txt"

with open(output, "w", encoding="utf-8") as fout:
    for f in [file1, file2]:
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8") as fin:
                fout.writelines(fin.readlines())

print(f"合併完成 -> {output}")