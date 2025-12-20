# from AI_CUP_predict_core import run_prediction

# PREDICT_MODEL = "./best84.pt"

# settings = {
#     "hist_eq": False,          # 是否開啟 Histogram Equalization
#     "batch_size": 32,           # 批次處理
#     "conf": 0.03,              # 預設 YOLO confidence threshold = 0.25
#     "iou": 0.5                # 預設 YOLO NMS IoU threshold = 0.45
# }

# #conf_list = [0.03, 0.05, 0.08, 0.10]
# run_prediction(PREDICT_MODEL, settings)
# # for conf in conf_list:
# #     settings["conf"] = conf
# #     run_prediction(PREDICT_MODEL, settings)


from AI_CUP_predict_core import run_prediction

models = [
    "./best103.pt",
    "./best104.pt",
    "./best105.pt",
    "./best106.pt",
    "./best107.pt"
]

settings = {
    "hist_eq": False,
    "batch_size": 32,
    "conf": 0.1,
    "iou": 0.5
}

for model in models:
    print(f"Running prediction with {model}")
    run_prediction(model, settings)
