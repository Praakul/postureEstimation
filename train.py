import torch
import datetime
import os
from torch.utils.data import DataLoader, random_split

# Modules
from argparser import get_config
from utils import set_seed, setup_logging
from dataset import SkeletonDataset
from model import STGCN
from trainer import Trainer

def main():
    # 1. Initialize Config
    cfg = get_config()
    set_seed(cfg.SEED)
    
    # Setup Run Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.RUN_NAME}_{timestamp}" if cfg.RUN_NAME else f"train_{timestamp}"
    run_dir = os.path.join(cfg.RUNS_DIR, run_name)
    
    logger = setup_logging(run_dir)
    logger.info(f"--- Training Started: {run_name} ---")
    logger.info(f"Settings: BS={cfg.BATCH_SIZE}, LR={cfg.LEARNING_RATE}, Augment={cfg.AUGMENT}")

    # 2. Load Data
    try:
        full_dataset = SkeletonDataset(
            cfg.X_PATH, 
            cfg.Y_PATH, 
            seq_length=cfg.SEQ_LENGTH, 
            augment=cfg.AUGMENT
        )
    except FileNotFoundError as e:
        logger.error(f"CRITICAL: {e}")
        return

    train_size = int(cfg.TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # Validation is always deterministic (no augmentation)
    val_data.dataset.augment = False 
    
    train_loader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    logger.info(f"Loaded {len(train_data)} training sequences and {len(val_data)} validation sequences.")

    # 3. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCN(
        num_classes=cfg.NUM_CLASSES, 
        in_channels=cfg.IN_CHANNELS
    ).to(device)
    
    logger.info(f"Model loaded on {device}.")

    # 4. Run Trainer
    trainer = Trainer(model, train_loader, val_loader, cfg, logger, device)
    trainer.fit(run_dir)
    
    logger.info(f"Done. Best Accuracy: {trainer.best_acc:.2f}%")

if __name__ == "__main__":
    main()