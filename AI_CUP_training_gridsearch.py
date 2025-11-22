import locale
import os
import shutil
import itertools
import pandas as pd

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# !pip install ultralytics
import ultralytics
ultralytics.checks()

#下載資料集

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

print("完成移動！")

from ultralytics import YOLO

# ============================================
# OPTION 1: QUICK GRID SEARCH (uncomment to run)
# ============================================
ENABLE_GRID_SEARCH = False  # Set to True to enable grid search

if ENABLE_GRID_SEARCH:
    print("\n" + "="*60)
    print("STARTING HYPERPARAMETER GRID SEARCH")
    print("="*60 + "\n")
    
    # Define parameter ranges
    lr0_list = [0.0005, 0.001, 0.005]
    dropout_list = [0.0, 0.05, 0.1]
    scale_list = [0.01, 0.05, 0.1]
    
    results_list = []
    
    # Iterate through all combinations
    total_runs = len(lr0_list) * len(dropout_list) * len(scale_list)
    current_run = 0
    
    for lr0, dropout, scale in itertools.product(lr0_list, dropout_list, scale_list):
        current_run += 1
        print(f"\n{'='*60}")
        print(f"Run {current_run}/{total_runs}: lr0={lr0}, dropout={dropout}, scale={scale}")
        print(f"{'='*60}\n")
        
        # Reload model for fresh start
        model = YOLO('yolov12n.pt')
        
        # Generate unique run name
        run_name = f"lr{lr0}_do{dropout}_sc{scale}".replace('.', 'p')
        
        # Train with this configuration
        results = model.train(
            data="./aortic_valve_colab.yaml",
            epochs=30,              # Use fewer epochs for grid search speed
            batch=8,
            imgsz=640,
            device=0,
            patience=5,             # Early stopping
            optimizer='AdamW',
            
            lr0=lr0,
            lrf=0.01,
            dropout=dropout,
            scale=scale,
            
            degrees=5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            
            cos_lr=True,
            
            project="runs/hyperparameter_search",
            name=run_name,
            exist_ok=True,
        )
        
        # Extract best mAP from results
        try:
            csv_path = f"runs/hyperparameter_search/{run_name}/results.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                best_map = df['metrics/mAP50'].max()
            else:
                best_map = 0
        except Exception as e:
            print(f"Could not extract mAP: {e}")
            best_map = 0
        
        # Store results
        results_list.append({
            'lr0': lr0,
            'dropout': dropout,
            'scale': scale,
            'best_mAP50': best_map,
            'run_name': run_name
        })
        
        print(f"Best mAP50 for this run: {best_map:.4f}\n")
    
    # ===== SUMMARY =====
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('best_mAP50', ascending=False)
    
    print("\n" + "="*60)
    print("GRID SEARCH SUMMARY (sorted by mAP50)")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Save to CSV
    results_df.to_csv("hyperparameter_search_results.csv", index=False)
    print("\nResults saved to: hyperparameter_search_results.csv")
    
    # Print best configuration
    best_row = results_df.iloc[0]
    print("\n" + "="*60)
    print("BEST CONFIGURATION:")
    print(f"  lr0={best_row['lr0']}")
    print(f"  dropout={best_row['dropout']}")
    print(f"  scale={best_row['scale']}")
    print(f"  Best mAP50={best_row['best_mAP50']:.4f}")
    print(f"  Run name: {best_row['run_name']}")
    print("="*60)

# ============================================
# OPTION 2: FINAL TRAINING (uncomment to run)
# ============================================
ENABLE_FINAL_TRAINING = True  # Set to True to enable final training

if ENABLE_FINAL_TRAINING:
    print("\n" + "="*60)
    print("STARTING FINAL TRAINING WITH OPTIMIZED PARAMETERS")
    print("="*60 + "\n")
    
    model = YOLO('yolov12n.pt')
    
    results = model.train(
        data="./aortic_valve_colab.yaml",
        epochs=15,                # Full training epochs
        batch=8,
        imgsz=640,
        device=0,
        patience=20,
        optimizer='AdamW',
        
        # === OPTIMIZED LEARNING RATE (adjust based on grid search results) ===
        lr0=0.001,                 # CHANGE THIS based on your grid search results
        lrf=0.01,                  # Final LR ratio
        
        # === OPTIMIZED AUGMENTATION PARAMETERS ===
        dropout=0.05,              # CHANGE THIS based on your grid search results
        scale=0.05,                # CHANGE THIS based on your grid search results
        
        # --- PRECISION MEDICAL AUGMENTATION ---
        degrees=5,                 # Small rotation (±5°)
        flipud=0.0,                # Vertical flip is BAD for hearts
        fliplr=0.5,                # Horizontal flip is OK
        
        # Drastically reduced "destructive" augmentations
        mosaic=0.0,                # Reduced from 1.0 (less stitching)
        mixup=0.0,                 # DISABLED: Blending images hurts medical precision
        copy_paste=0.0,            # DISABLED: Floating objects confuse anatomy
        
        # Refinement Strategy
        cos_lr=True,
        close_mosaic=50,           # Train on REAL images for the last 50 epochs
    )
    
    print("\n" + "="*60)
    print("FINAL TRAINING COMPLETE!")
    print("Best model saved as: ./best.pt")
    print("="*60)