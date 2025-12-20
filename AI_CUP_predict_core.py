import os
import zipfile
import shutil
import cv2
import torch
from ultralytics import YOLO


# ----------------------------------------
# Utility: extract model name for output
# "./best48.pt" ??"best48"
# ----------------------------------------
def model_name_to_output(model_path, conf):
    base = os.path.basename(model_path)      # "best48.pt"
    name = os.path.splitext(base)[0]         # "best48"
    return f"merged_{name}_conf_{conf:.2f}.txt"



# ----------------------------------------
# Histogram Equalization
# ----------------------------------------
def apply_hist_eq(img):
    # Apply CLAHE on each channel for medical images
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return final


# ----------------------------------------
# Unzip + prepare folder
# ----------------------------------------
def prepare_directories():
    dirs = [
        "./datasets/test/tmp",
        "./datasets/test/images1",
        "./datasets/test/images2",
        "./predict_txt"
    ]
    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)


# use it when needed
def unzip_testing_zip():
    zip_file = "testing_image.zip"
    target = "./datasets/test/tmp"

    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(target)
        print(f"Unzipped {zip_file} to {target}")
    else:
        print(f"Error: {zip_file} not found!")


# ----------------------------------------
# Split images into images1 and images2
# ----------------------------------------
def split_images():
    base_root = "./datasets/test/tmp"
    dst1 = "./datasets/test/images1"
    dst2 = "./datasets/test/images2"

    # Find patient root
    patient_root = base_root
    for dirpath, dirnames, _ in os.walk(base_root):
        if any(d.lower().startswith("patient") for d in dirnames):
            patient_root = dirpath
            break

    all_files = []
    for patient_folder in os.listdir(patient_root):
        current = os.path.join(patient_root, patient_folder)
        if os.path.isdir(current) and patient_folder.lower().startswith("patient"):
            for fname in os.listdir(current):
                if fname.lower().endswith(".png"):
                    all_files.append(os.path.join(current, fname))

    all_files.sort()
    half = len(all_files) // 2

    for f in all_files[:half]:
        shutil.move(f, os.path.join(dst1, os.path.basename(f)))

    for f in all_files[half:]:
        shutil.move(f, os.path.join(dst2, os.path.basename(f)))

    print(f"Split total {len(all_files)} images ??first {half}, second {len(all_files)-half}")


# ----------------------------------------
# Predict for folder & write txt
# ----------------------------------------
def predict_folder(model, folder_path, output_txt, settings):
    img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".png")])
    total = len(img_files)
    results_count = 0

    if total == 0:
        print(f"No images found in {folder_path}")
        return 0

    print(f"Predicting {total} images in {folder_path} ...")

    best_preds = {}

    for start in range(0, total, settings["batch_size"]):
        batch = img_files[start:start + settings["batch_size"]]
        imgs = []

        print(f"Progress: {start + 1} / {total}  ({((start + 1) / total) * 100:.1f}%)")

        # Load and preprocess each image
        for img_name in batch:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            if settings["hist_eq"]:
                img = apply_hist_eq(img)

            imgs.append(img)

        conf_threshold = settings["conf"] if settings["conf"] is not None else 0.25

        results = model.predict(
            source=imgs,
            conf=conf_threshold,
            iou=settings["iou"],
            save=False,
            imgsz=640,
            device=0
        )

        results_count += len(results)

        for i, res in enumerate(results):
            filename = batch[i].replace(".png", "")
            boxes = res.boxes
            n = len(boxes.cls.tolist())

            if n > 0:
                for j in range(n):
                    cls = int(boxes.cls[j].item())
                    conf = float(boxes.conf[j].item())
                    x1, y1, x2, y2 = boxes.xyxy[j].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    if filename not in best_preds or conf > best_preds[filename]["conf"]:
                        best_preds[filename] = {
                            "cls": cls,
                            "conf": conf,
                            "coords": (x1, y1, x2, y2)
                        }

    with open(output_txt, "w") as f_output:
        for filename in sorted(best_preds.keys()):
            pred = best_preds[filename]
            x1, y1, x2, y2 = pred["coords"]
            line = f"{filename} {pred['cls']} {pred['conf']:.4f} {x1} {y1} {x2} {y2}\n"
            f_output.write(line)

    print(f"Finished predicting {total} images in {folder_path}.")
    torch.cuda.empty_cache()
    return results_count



# ----------------------------------------
# Merge txt files
# ----------------------------------------
def merge_txt(output_filename):
    file1 = "./predict_txt/images1.txt"
    file2 = "./predict_txt/images2.txt"
    output = f"./predict_txt/{output_filename}"

    with open(output, "w", encoding="utf-8") as fout:
        for fpath in [file1, file2]:
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as fin:
                    fout.writelines(fin.readlines())

    print(f"合併完成 → {output}")
    return output


# ----------------------------------------
# MAIN ENTRY
# ----------------------------------------
def run_prediction(PREDICT_MODEL, settings):
    print(f"\n=== Using model: {PREDICT_MODEL} ===")
    print("Settings:", settings)

    prepare_directories()
    # unzip_testing_zip() # 
    split_images()

    model = YOLO(PREDICT_MODEL)

    print("\nPredicting images1...")
    predict_folder(model, "./datasets/test/images1", "./predict_txt/images1.txt", settings)

    print("\nPredicting images2...")
    predict_folder(model, "./datasets/test/images2", "./predict_txt/images2.txt", settings)

    conf = settings["conf"]
    output_name = model_name_to_output(PREDICT_MODEL, conf)
    merge_txt(output_name)


    print("\n=== Prediction Done ===")
