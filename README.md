# AI CUP - 電腦視覺期末報告

## 專案簡介

本專案為電腦視覺期末競賽的研究報告，主要針對醫療影像中的主動脈瓣（Aortic Valve）進行物件偵測。我們使用 YOLOv12 模型作為基礎，並嘗試多種改進方法，包括資料前處理、模型架構調整、以及集成學習等技術，以提升偵測準確率。

### 貢獻者
- 洪理川、王宸澤
---

## 目錄
1. [實驗方法](#實驗方法)
   - [超參數設定與資料前處理](#超參數設定與資料前處理)
   - [模型架構調整](#模型架構調整)
2. [實驗結果](#實驗結果)
3. [討論與結論](#討論與結論)
4. [參考文獻](#參考文獻)

---

## 實驗方法

### 超參數設定與資料前處理

#### 1. Augmentation

我們嘗試對每張圖片進行以下增強操作：
- Rotation（旋轉）
- Translation（平移）
- Horizontal flip（水平翻轉）
- Scaling（縮放）

**實驗結果：**
將 testing data 的預測結果顯示，進行 augmentation 反而使 mAP50 降低至 0.85067698 到 0.90168443 之間。

**原因分析：**
推測原因為本次競賽提供的資料是某位患者在一次治療中，連續一段時間內的電腦斷層影像，大部分資料都長得很像。使用 augmentation 讓模型學習各種多樣狀況，反而不利於此競賽的表現。因此後續實驗不再採用這種 augmentation。

#### 2. 使用不同版本的 YOLOv12

**比較版本：**
- Ultralytics 官方版本 YOLOv12
- 論文版本 YOLOv12（Y. Tian et al. [2]）

**架構差異：**
- Ultralytics 版本：約 272 layers（以 nano 模型為例）
- 論文版本：約 497 layers（深度增加 80%～90%）

**實驗結果：**
論文版本的預測準確率反而不如 Ultralytics 版本。推測在本競賽資料較少的條件下，過深的模型反而較不利於穩定訓練與提升表現。

**結論：**
後續實驗皆採用 **Ultralytics 原版 YOLOv12** 作為主要的影像辨識模型。

#### 3. 使用不同參數量的 YOLOv12

**測試模型：**
- YOLOv12n（nano）
- YOLOv12s（small）
- YOLOv12m（medium）

**實驗結果：**
參數量位於中間的 **YOLOv12m** 有最佳的辨識準確率，在 public leaderboard 達到 **0.94263301**。

**原因分析：**
- 資料集僅有 16,863 張圖片
- 參數量最大的模型：樣本數不夠 → underfitting
- 參數量較小的模型：彈性不夠 → overfitting
- 規模中等的 YOLOv12m 較適合本競賽資料集

#### 4. Train/Validation 資料比例調整

**測試比例：**（以病人數量區分）
- 30:20
- 40:10
- 49:1

**實驗結果：**
- **40:10** 比例在 public leaderboard 有最好表現
- **49:1** 比例在 private leaderboard 有較好表現
- 兩者的 mAP50 都能達到 **0.94**
- **30:20** 比例的 mAP50 相較減少 0.01~0.02

**結論：**
由於資料集較小，提供更多訓練資料能讓模型有較好的成效。

#### 5. 讓模型學習「沒有主動脈瓣」的圖片

**背景：**
資料集中約有 80% 的圖片不含主動脈瓣。

**實驗設定：**
1. 將所有影像都拿來訓練（含有/不含主動脈瓣）
2. 取具有主動脈瓣影像數量的 20% 的不含主動脈瓣影像一起訓練
3. Baseline：只取具有主動脈瓣的影像訓練

**實驗結果：**
不使用任何不含主動脈瓣影像的訓練資料仍然有**最高準確率**，將所有圖片一起訓練的版本有**最差表現**。

**原因分析：**
含有主動脈瓣的影像資料量不足，加入不含主動脈瓣的影像訓練，反而讓模型無法準確學習具有主動脈瓣的影像特徵。

#### 6. 另行訓練 Classifier 先行分類 Test Data

**方法：**
1. 使用 YOLOv11n-cls 訓練分類器
2. 將 training data 分類為 positive（存在主動脈瓣）和 negative（不存在主動脈瓣）
3. 使用物件偵測模型對被分類為 positive 的 test data 進行物件位置判別

**實驗結果：**
對資料 classify 後，有時能提高準確率，有時會降低準確率。因此在後續實驗中，部分會再嘗試是否要進行 classify 操作。

#### 7. Histogram Equalization

我們使用 OpenCV 的 **CLAHE**（Contrast Limited Adaptive Histogram Equalization）來提升 CT 影像的局部對比。

**設定參數：**
- Clip limit = 2.0
- Tile grid size = 8×8

**實驗一：作為資料前處理（Data Preprocessing）**
- 結果：public leaderboard 準確率由 0.93713821 降至 0.90168443

**實驗二：作為資料增強（Data Augmentation）**
- 結果：training validation 上最佳 mAP 僅有 0.9222，明顯低於 baseline 的 0.969

**結論：**
最終決定完全移除 histogram equalization（CLAHE）相關的前處理與 augmentation。

#### 8. K-fold Validation 與模型的 Ensembling

**方法：**
- 採用 **5-fold validation** 訓練，得到 5 個模型
- 使用 **Weighted Box Fusion (WBF)** 將多個模型的 bounding box 融合

**Weighted Box Fusion 原理：**
1. 針對同一張影像、同一類別的預測框，依據 IoU 是否大於門檻（例如 IoU ≥ 0.5）進行分組
2. 同一組的 bounding box 進行融合，座標以加權平均方式計算
3. 權重與 confidence 成正比，使高 confidence 的預測具有較大影響

**實驗結果：**
加入 K-fold ensemble 後，模型有明顯提升，其中 **YOLOv12m + K-fold ensemble** 的 private leaderboard 可達 **0.9585**，是本次實驗中準確率最高。

---

### 模型架構調整

#### 1. 將 YOLOv12 的 Backbone 改為 ConvNeXt

**實作方法：**
1. 修改 yolo12.yaml 檔案，將 backbone 部分替換成 TorchVision 提供的 ConvNeXt 模型
2. 使用預訓練權重，不需額外使用 COCO dataset 進行預訓練
3. 從 ConvNeXt backbone 的不同 stage 輸出 feature map，作為 P3、P4、P5 feature
4. 使用 FPN 向上採樣和 PAN 向下採樣結合語意和位置資訊

**實驗結果：**
- 導入 20% 的不含主動脈瓣影像訓練
- Public leaderboard 的 mAP 達到 **0.94371780**
- 相比原版 YOLOv12 模型有小幅提升

#### 2. 將 YOLOv12 的 Neck 改為 BiFPN + CBAM

**實作方法：**
1. 自行撰寫客製化的 BiFPN 模組（置於 `ultralytics/nn/modules/bifpn.py`）
2. 在 `ultralytics/nn/modules/__init__.py` 中註冊模組
3. 使用 1×1 convolution 將 P3、P4、P5 的通道數統一調整為 256 channels
4. 將 BiFPN 的三個輸出特徵接上 CBAM 進行注意力強化

**架構特點：**
- 簡化版 BiFPN（相較於 EfficientDet 論文版本）
- 每個融合節點的輸入較少（多為兩路融合）
- 採用 P3～P5 三個尺度

**實驗結果：**
修改成 BiFPN + CBAM 後的模型表現仍與原本 YOLOv12m 有一定差距，public leaderboard 的成績約為 **0.910**，未能如預期提升整體 mAP50 表現。

---

## 實驗結果

### 1. Augmentation 實驗結果

| 方法 | Public Leaderboard | Private Leaderboard |
|------|-------------------|---------------------|
| 無 Augmentation | 更高 | 更高 |
| 有 Augmentation | 0.85067698 - 0.90168443 | - |

### 2. 不同版本 YOLOv12 實驗結果

| 模型版本 | 架構特點 | 表現 |
|---------|---------|------|
| Ultralytics YOLOv12m | ~272 layers | 較佳 |
| 論文版 YOLOv12m | ~497 layers | 較差 |

### 3. 不同參數量 YOLOv12 實驗結果

| 模型 | Public Leaderboard |
|------|-------------------|
| YOLOv12n | - |
| YOLOv12m | **0.94263301** |

### 4. Train/Validation 比例實驗結果

| 比例 | Public Leaderboard | Private Leaderboard |
|------|-------------------|---------------------|
| 30:20 | ~0.92-0.93 | ~0.92-0.93 |
| 40:10 | **~0.94** | ~0.94 |
| 49:1 | ~0.94 | **~0.94** |

### 5. K-fold Ensemble 實驗結果

| 模型 | Public Leaderboard | Private Leaderboard |
|------|-------------------|---------------------|
| YOLOv12m | 0.94263301 | - |
| YOLOv12m + K-fold Ensemble | - | **0.9585** ⭐ |

### 6. 模型架構調整實驗結果

| 架構 | Public Leaderboard | Private Leaderboard |
|------|-------------------|---------------------|
| YOLOv12m | 0.94263301 | - |
| ConvNeXt Backbone | **0.94371780** | - |
| ConvNeXt + BiFPN + CBAM | ~0.910 | - |

---

## 討論與結論

### 主要發現

1. **YOLOv12m 為最佳基礎模型**：對於本競賽的電腦斷層醫療影像，使用 YOLOv12m 模型就能得到優秀的結果。

2. **K-fold Ensemble 效果顯著**：使用 K-fold ensemble 來整合多個模型的結果後，能夠得到更好的成效，private leaderboard 可達 **0.9585**。

3. **資料增強不適用**：由於競賽資料為連續的電腦斷層影像，傳統的資料增強方法反而降低準確率。

4. **架構調整效果有限**：雖然嘗試了 ConvNeXt、BiFPN、CBAM 等先進架構，但對結果影響不大，甚至部分方法降低了準確率。

### 最佳模型表現

**競賽期間最佳成績：**
- **Public Leaderboard**: YOLOv12m（mAP50: 0.94263301）
- **Private Leaderboard**: YOLOv12m with optimal train/val ratio

**競賽結束後最佳成績：**
- **YOLOv12m + K-fold Ensemble**（Private Leaderboard: **0.9585**）

### 未來方向

未來若有機會再參與相關競賽，可以：
1. 優先觀看專注於「電腦斷層影像物件偵測」的相關研究
2. 從這些研究中思考如何調整架構與實驗方式
3. 針對醫療影像的特性設計更合適的前處理方法

---

## 參考文獻

[1] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in CVPR, 2022.

[2] Y. Tian, Q. Ye, and D. Doermann, "YOLOv12: Attention-Centric Real-Time Object Detectors," arXiv preprint arXiv:2502.12524, Feb. 2025.

[3] M. Tan, R. Pang, and Q. V. Le, "EfficientDet: Scalable and Efficient Object Detection," in CVPR, 2020.

[4] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," in ECCV, 2018.

[5] J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," in CVPR, 2018, pp. 7132–7141.

[6] H. Tekin, Ş. Kılıç, and Y. Doğan, "DiagNeXt: A Two-Stage Attention-Guided ConvNeXt Framework for Kidney Pathology Segmentation and Classification," J. Imaging, vol. 11, no. 12, Art. no. 433, Dec. 2025.

[7] A. Roy et al., "MedNeXt: Transformer-Driven Scaling of ConvNets for Medical Image Segmentation," in MICCAI, 2023.

[8] R. Solovyev, W. Wang, and T. Gabruseva, "Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models," arXiv preprint arXiv:1910.13302, 2019.

---

## 專案結構

```
AI_CUP/
├── README.md                                    # 本文件
├── 電腦視覺期末報告.docx                         # 原始報告文件
├── AI_CUP_training.py                           # 主要訓練腳本
├── AI_CUP_predict.py                            # 預測腳本
├── AI_CUP_classify_train.py                     # 分類器訓練腳本
├── AI_CUP_classify_predict.py                   # 分類器預測腳本
├── ensemble.py                                  # 集成學習腳本
├── weigted_box_fusion.py                        # WBF 實作
├── yolov12/                                     # YOLOv12 模型目錄
└── *.pt                                         # 訓練好的模型權重
```

---

## 使用方法

### 環境設置

```bash
pip install ultralytics
```

### 訓練模型

```bash
python AI_CUP_training.py
```

### 進行預測

```bash
python AI_CUP_predict.py
```

### 使用 K-fold Ensemble

```bash
# 訓練 K-fold 模型
python AI_CUP_training_kfold.py

# 進行集成預測
python ensemble.py
```

---

## 致謝

感謝競賽主辦方提供資料集與競賽平台，以及在競賽期間給予的技術支援。
