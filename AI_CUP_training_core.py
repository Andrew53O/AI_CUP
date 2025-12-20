"""
AI_CUP_training_core.py
Clean training pipeline for AI CUP Aortic Valve Detection.

Supports:
- histogram equalization (train only)
- include unlabeled images
- train/val split (shuffle or sequential)
- train_data_mode: "original", "hist", "both"
- augmentation on/off
- patience in settings
- auto-generate train_config.yaml
"""

import os
import shutil
import random
import cv2
import yaml
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import metrics


# =============================================================================
# Utility: ensure folder exists and is empty
# =============================================================================
def ensure_clean(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# =============================================================================
# Histogram equalization
# =============================================================================
def apply_hist_eq(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# =============================================================================
# Move labeled images and collect unlabeled
# =============================================================================
def move_labeled(patient, IMG_ROOT, LBL_ROOT, split):
    img_dir = os.path.join(IMG_ROOT, patient)
    lbl_dir = os.path.join(LBL_ROOT, patient)
    unlabeled = []

    if not os.path.isdir(img_dir):
        return unlabeled

    for fname in os.listdir(img_dir):
        if not fname.endswith(".png"):
            continue

        img_path = os.path.join(img_dir, fname)
        base, _ = os.path.splitext(fname)
        label_path = os.path.join(lbl_dir, base + ".txt")

        if os.path.exists(label_path):
            shutil.copy2(img_path, f"./datasets/{split}/images/")
            shutil.copy2(label_path, f"./datasets/{split}/labels/")
        else:
            unlabeled.append(base)

    return unlabeled


# =============================================================================
# Move unlabeled images
# =============================================================================
def move_unlabeled(patient, unl_list, IMG_ROOT, split):
    if len(unl_list) == 0:
        return

    img_dir = os.path.join(IMG_ROOT, patient)

    if len(unl_list) < 10:
        sample_list = unl_list
    else:
        step = len(unl_list) // 10
        sample_list = unl_list[::step]

    for base in sample_list:
        img_path = os.path.join(img_dir, base + ".png")
        shutil.copy2(img_path, f"./datasets/{split}/images/")
        open(f"./datasets/{split}/labels/{base}.txt", "w").close()


# =============================================================================
# Generate histogram-equalized training folder
# =============================================================================
def build_hist_folder():
    src_img = "./datasets/train/images"
    src_lbl = "./datasets/train/labels"
    dst_img = "./datasets/train_hist/images"
    dst_lbl = "./datasets/train_hist/labels"

    for fname in os.listdir(src_img):
        if fname.endswith(".png"):
            img = cv2.imread(os.path.join(src_img, fname))
            eq = apply_hist_eq(img)
            cv2.imwrite(os.path.join(dst_img, fname), eq)

            base, _ = os.path.splitext(fname)
            shutil.copy2(os.path.join(src_lbl, base + ".txt"), dst_lbl)


# =============================================================================
# Build train/val sets
# =============================================================================
def build_dataset(settings, IMG_ROOT, LBL_ROOT):
    # Sequential or shuffle patient list
    if settings["shuffle_patients"]:
        patients = sorted([p for p in os.listdir(IMG_ROOT) if p.startswith("patient")])
        random.shuffle(patients)
        n = len(patients)
        train_n = int(n * settings["train_ratio"])
        train_list = patients[:train_n]
        val_list = patients[train_n:]
    else:
        a, b = settings["train_range"]
        c, d = settings["val_range"]
        train_list = [f"patient{i:04d}" for i in range(a, b + 1)]
        val_list = [f"patient{i:04d}" for i in range(c, d + 1)]

    # Move train
    for p in train_list:
        unl = move_labeled(p, IMG_ROOT, LBL_ROOT, "train")
        if settings["include_unlabeled"]:
            move_unlabeled(p, unl, IMG_ROOT, "train")

    # Histogram (train only)
    if settings["hist_eq"]:
        build_hist_folder()

    # Move val
    for p in val_list:
        unl = move_labeled(p, IMG_ROOT, LBL_ROOT, "val")
        if settings["include_unlabeled"]:
            move_unlabeled(p, unl, IMG_ROOT, "val")


# =============================================================================
# Custom fitness (mAP50 only)
# =============================================================================
def custom_fitness(self):
    w = [0.0, 0.0, 1.0, 0.0]  # only mAP50
    return (np.array(self.mean_results()) * w).sum()

metrics.Metric.fitness = custom_fitness


# =============================================================================
# Save best metrics
# =============================================================================
def save_best(trainer):
    m = getattr(trainer, "metrics", None)
    if not m:
        return
    path = Path(trainer.save_dir) / "best_metrics.txt"
    with open(path, "w") as f:
        f.write("BEST.PT METRICS\n")
        for k in [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)"
        ]:
            if k in m:
                f.write(f"{k}: {m[k]:.4f}\n")
    print(f"Saved best_metrics.txt → {path}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def run_training(settings):
    global IMG_ROOT, LBL_ROOT

    IMG_ROOT = settings["IMG_ROOT"]
    LBL_ROOT = settings["LBL_ROOT"]

    # clean dataset folders
    ensure_clean("./datasets/train/images")
    ensure_clean("./datasets/train/labels")
    ensure_clean("./datasets/val/images")
    ensure_clean("./datasets/val/labels")

    if settings["hist_eq"]:
        ensure_clean("./datasets/train_hist/images")
        ensure_clean("./datasets/train_hist/labels")

    # build dataset
    build_dataset(settings, IMG_ROOT, LBL_ROOT)

    # determine train paths
    mode = settings["train_data_mode"]

    if mode == "original":
        train_paths = "./datasets/train/images"

    elif mode == "hist":
        train_paths = "./datasets/train_hist/images"

    elif mode == "both":
        if settings["hist_eq"]:
            train_paths = [
                "./datasets/train/images",
                "./datasets/train_hist/images"
            ]
        else:
            train_paths = "./datasets/train/images"
            print("WARNING: train_data_mode='both' but hist_eq=False. Using original only.")
    else:
        raise ValueError("train_data_mode must be 'original', 'hist', or 'both'.")

    # FIXED YAML GENERATION 
    BASE = os.path.abspath(".")

    def abs_path(p):
        # Convert "./datasets/train/images" → "/home/.../datasets/train/images"
        return os.path.join(BASE, p.replace("./", "", 1))

    # Handle 'both' mode (list) or single path
    if isinstance(train_paths, list):
        yaml_train = [abs_path(p) for p in train_paths]
    else:
        yaml_train = abs_path(train_paths)

    yaml_val = abs_path("./datasets/val/images")

    yaml_dict = {
        "train": yaml_train,
        "val": yaml_val,
        "nc": 1,
        "names": ["aortic_valve"]
    }

    yaml_path = "./datasets/train_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

    print("\nGenerated YAML:")
    print(yaml_dict)

    # start training
    model = YOLO(settings["model"])
    model.add_callback("on_train_end", save_best)

    model.train(
        data=yaml_path,
        epochs=settings["epochs"],
        batch=settings["batch"],
        imgsz=settings["imgsz"],
        device=settings["device"],
        patience=settings["patience"],
        **(settings["augment_params"] if settings["augment"] else {})
    )
