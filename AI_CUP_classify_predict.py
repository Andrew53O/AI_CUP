from ultralytics import YOLO
import os
import cv2
import torch
import shutil

POSITIVE_ROOT = "./datasets_cls/test/positive/"
NEGATIVE_ROOT = "./datasets_cls/test/negative/"

os.makedirs(POSITIVE_ROOT, exist_ok=True)
os.makedirs(NEGATIVE_ROOT, exist_ok=True)

def predict_folder(model, folder_path, settings):
    img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".png")])
    total = len(img_files)
    results_count = 0

    if total == 0:
        print(f"No images found in {folder_path}")
        return 0

    print(f"Predicting {total} images in {folder_path} ...")


    for start in range(0, total, settings["batch_size"]):
        batch = img_files[start:start + settings["batch_size"]]
        imgs = []

        print(f"Progress: {start + 1} / {total}  ({((start + 1) / total) * 100:.1f}%)")

        # Load and preprocess each image
        for img_name in batch:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            imgs.append(img)


        results = model.predict(
            source=imgs,
            save=False,
            imgsz=640,
            device=0
        )

        results_count += len(results)

        for i, r in enumerate(results):
            cls_idx = int(r.probs.top1)
            cls_name = r.names[cls_idx]
            conf = float(r.probs.top1conf)
            filename = batch[i]
            filename = os.path.join(folder_path, filename)
            # print("filename=",filename)
            if cls_name == "positive" and conf >= settings["cls_conf_threshold"]:
                shutil.copy2(filename, POSITIVE_ROOT)
            else:
                shutil.copy2(filename, NEGATIVE_ROOT)

    print(f"Finished predicting {total} images in {folder_path}.")
    torch.cuda.empty_cache()
    return results_count

# ----------------------------------------
# MAIN ENTRY
# ----------------------------------------
def run_prediction(PREDICT_MODEL, settings):
    print(f"\n=== Using model: {PREDICT_MODEL} ===")
    print("Settings:", settings)

    model = YOLO(PREDICT_MODEL)

    print("\nPredicting images1...")
    predict_folder(model, "./datasets/test/images1", settings)

    print("\nPredicting images2...")
    predict_folder(model, "./datasets/test/images2", settings)

    print("\n=== Prediction Done ===")

if __name__ == "__main__":
    PREDICT_MODEL = "./runs/classify/train6/weights/best.pt"
    settings = {
        "batch_size": 32,
        "cls_conf_threshold": 0.6
    }
    run_prediction(PREDICT_MODEL, settings)