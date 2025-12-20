from AI_CUP_training_core import run_training

settings = {
    "model": "yolo12m.pt",

    # dataset roots
    "IMG_ROOT": "./training_image",
    "LBL_ROOT": "./training_label",

    # train/val split
    "shuffle_patients": False,
    # "train_ratio": 0.8,
    # "val_ratio": 0.2,

    # sequential fallback
    "train_range": (1, 40),
    "val_range": (41, 50),

    # unlabeled control
    "include_unlabeled": False,

    # histogram eq
    "hist_eq": True,

    # what to train on: original / hist / both
    "train_data_mode": "hist",

    # augmentation
    "augment": False,
    # "augment_params": {
    #     "flipud": 0.0,
    #     "fliplr": 0.5,
    #     "hsv_h": 0.015,
    #     "hsv_s": 0.7,
    #     "hsv_v": 0.4,
    #     "scale": 0.5,
    # },

    # training
    "batch": 16,
    "epochs": 80,
    "patience": 20,
    "device": 0,
    "imgsz": 640
}

run_training(settings)
