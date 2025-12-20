
import os
from ensemble import *

# THIS PART BELOW CAN BE CHANGED
IMG_WIDTH  = 640
IMG_HEIGHT = 640


prediction_list = [
    "predict_txt_last/merged_best103_conf_0.10.txt",
    "predict_txt_last/merged_best104_conf_0.10.txt",
    "predict_txt_last/merged_best105_conf_0.10.txt",
    "predict_txt_last/merged_best106_conf_0.10.txt",
    "predict_txt_last/merged_best107_conf_0.10.txt",
]

IOU             = 0.5
CONF            = 0.03
output_file     = 'predict_txt_last/ensembled_prediction_modelCBC-kfold_5_no_classify_conf_0.10.txt'
# END OF CHANGEABLE PART

fold_weights    = [1 for i in range(len(prediction_list))]
all_predictions = []

for i, file_path in enumerate(prediction_list):
    pred = read_merged_txt_file(file_path, IMG_WIDTH, IMG_HEIGHT)
    all_predictions.append(pred)

result = ensemble_all_predictions(
    all_predictions, 
    weights      = fold_weights, 
    iou_thr      = IOU, 
    skip_box_thr = CONF
)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
save_predictions(result, output_file, IMG_WIDTH, IMG_HEIGHT)
print(f"Saved final ensembled predictions to \'{output_file}'")