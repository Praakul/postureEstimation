import argparse
import os
from dataclasses import dataclass

# --- 1. DEFAULT SETTINGS (Formerly config.py) ---
DEFAULTS = {
    # Paths
    "DATA_DIR": ".",
    "X_PATH": "X_train.npy",
    "Y_PATH": "y_train.npy",
    "RUNS_DIR": "runs",
    "MODEL_SAVE_NAME": "stgcn_posture_model.pth",
    
    # Hyperparameters
    "BATCH_SIZE": 32,
    "EPOCHS": 50,
    "LR": 0.1,
    "WEIGHT_DECAY": 0.0005,
    "LR_MILESTONES": [20, 40],
    "LR_GAMMA": 0.1,
    "LABEL_SMOOTHING": 0.1,
    "EARLY_STOPPING_PATIENCE": 15,
    
    # Data & Model
    "NUM_CLASSES": 3,
    "IN_CHANNELS": 6, # 3 Pos + 3 Vel
    "SEQ_LENGTH": 50,
    "TRAIN_SPLIT": 0.8,
    "SEED": 42
}

@dataclass
class Config:
    """Runtime configuration object passed to Trainer/Dataset."""
    # We define types here for autocomplete, values are populated dynamically
    DATA_DIR: str
    X_PATH: str
    Y_PATH: str
    RUNS_DIR: str
    MODEL_SAVE_NAME: str
    BATCH_SIZE: int
    EPOCHS: int
    LEARNING_RATE: float
    WEIGHT_DECAY: float
    LR_MILESTONES: list
    LR_GAMMA: float
    LABEL_SMOOTHING: float
    EARLY_STOPPING_PATIENCE: int
    NUM_CLASSES: int
    IN_CHANNELS: int
    SEQ_LENGTH: int
    TRAIN_SPLIT: float
    SEED: int
    AUGMENT: bool = True
    RUN_NAME: str = None

def get_config():
    """
    Parses CLI arguments and returns a populated Config object.
    """
    parser = argparse.ArgumentParser(
        description="Industrial Safety ST-GCN Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment Group
    exp = parser.add_argument_group('Experiment')
    exp.add_argument("--name", type=str, default=None, help="Experiment run name")
    exp.add_argument("--seed", type=int, default=DEFAULTS["SEED"], help="Random seed")
    exp.add_argument("--runs_dir", type=str, default=DEFAULTS["RUNS_DIR"], help="Log directory")

    # Hyperparameters Group
    hparam = parser.add_argument_group('Hyperparameters')
    hparam.add_argument("--batch_size", type=int, default=DEFAULTS["BATCH_SIZE"], help="Batch size")
    hparam.add_argument("--epochs", type=int, default=DEFAULTS["EPOCHS"], help="Training epochs")
    hparam.add_argument("--lr", type=float, default=DEFAULTS["LR"], help="Learning rate")
    hparam.add_argument("--weight_decay", type=float, default=DEFAULTS["WEIGHT_DECAY"], help="Weight decay")
    hparam.add_argument("--label_smoothing", type=float, default=DEFAULTS["LABEL_SMOOTHING"], help="Label smoothing")

    # Data/Model Group
    data = parser.add_argument_group('Data & Model')
    data.add_argument("--seq_len", type=int, default=DEFAULTS["SEQ_LENGTH"], help="Sequence length")
    data.add_argument("--in_channels", type=int, default=DEFAULTS["IN_CHANNELS"], help="Input channels")
    data.add_argument("--no_augment", action="store_true", help="Disable data augmentation")

    args = parser.parse_args()

    # Create and populate Config object
    cfg = Config(
        DATA_DIR=DEFAULTS["DATA_DIR"],
        X_PATH=DEFAULTS["X_PATH"],
        Y_PATH=DEFAULTS["Y_PATH"],
        RUNS_DIR=args.runs_dir,
        MODEL_SAVE_NAME=DEFAULTS["MODEL_SAVE_NAME"],
        
        BATCH_SIZE=args.batch_size,
        EPOCHS=args.epochs,
        LEARNING_RATE=args.lr,
        WEIGHT_DECAY=args.weight_decay,
        LR_MILESTONES=DEFAULTS["LR_MILESTONES"],
        LR_GAMMA=DEFAULTS["LR_GAMMA"],
        LABEL_SMOOTHING=args.label_smoothing,
        EARLY_STOPPING_PATIENCE=DEFAULTS["EARLY_STOPPING_PATIENCE"],
        
        NUM_CLASSES=DEFAULTS["NUM_CLASSES"],
        IN_CHANNELS=args.in_channels,
        SEQ_LENGTH=args.seq_len,
        TRAIN_SPLIT=DEFAULTS["TRAIN_SPLIT"],
        SEED=args.seed,
        
        AUGMENT=not args.no_augment,
        RUN_NAME=args.name
    )
    
    return cfg