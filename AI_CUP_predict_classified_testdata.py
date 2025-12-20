from AI_CUP_predict_core_classified_testdata import run_prediction

models = [
    "./best99.pt",
    "./best101.pt"
]

settings = {
    "hist_eq": False,          # 是否開啟 Histogram Equalization
    "batch_size": 32,           # 批次處理
    "conf": 0.03,              # 預設 YOLO confidence threshold = 0.25
    "iou": 0.5,                # 預設 YOLO NMS IoU threshold = 0.45
    "dataset_path": "./datasets_cls/test/positive/"  # 測試集資料夾路徑
}


for model in models:
    print(f"Running prediction with {model}")
    run_prediction(model, settings)
