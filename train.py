import os
import sys
import yaml

import torch
torch.autograd.set_detect_anomaly(False)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
from einops import rearrange
from PIL import Image
import numpy as np
import random

from torch.optim.lr_scheduler import CosineAnnealingLR

from backbones import BackboneWrapper
from modules import (
    Detector,
    ClassAwareAugmentation,
    FocalLoss, DiceLoss,
    resize_keep_ratio_short_side as resize_img
)

# Unified dataset interface
from datasets import get_dataset, build_caa_category_dict


# ==============================================================================
# Load training config
# ==============================================================================
def load_config(config_path="config_train.yaml"):
    """Load the training config from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

# ==============================================================================
# Fix the random seed
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============== Lock data shuffling during training =================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(config=config):   
    seed_value      = config["seed_value"]
    set_seed(seed_value)

    
    backbone_name   = config["backbone"]
    dataset         = config["dataset"]

    batch_size      = config["batch_size"]
    num_workers     = config["num_workers"]
    
    lr              = config["lr"]
    eta_min         = config["cos_eta_min"]

    cls_wd          = config["cls_wd"]
    seg_wd          = config["seg_wd"]
    
    logits_fusion   = config["logits_fusion"]
    
    epochs          = config["epochs"]
    ckpt_save_epoch = config["ckpt_save_epoch"]
    checkpoint_dir  = config["checkpoint_dir"]

    # ---- Automatically select image size and normalization parameters based on backbone type ----
    if backbone_name == "DINO":
        img_size  = BackboneWrapper.DINO_IMG_SIZE
        
        mean      = BackboneWrapper.DINO_MEAN
        std       = BackboneWrapper.DINO_STD
    else:
        img_size  = BackboneWrapper.CLIP_IMG_SIZE
        
        mean      = BackboneWrapper.CLIP_MEAN
        std       = BackboneWrapper.CLIP_STD
    
    config["img_size"] = img_size

    def resize_img_bilinear(img):
        return resize_img(img, img_size=img_size, interpolation=Image.BILINEAR)

    def resize_mask_nearest(mask):
        return resize_img(mask, img_size=img_size, interpolation=Image.NEAREST)

    img_transform = transforms.Compose([
        transforms.Lambda(resize_img_bilinear),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    mask_transform = transforms.Compose([
        transforms.Lambda(resize_mask_nearest),
        transforms.CenterCrop(img_size),
    ])

    # ================= Automatically generate cache_name: {dataset}_{backbone}_{img_size}.pth =================
    cache_name = f"{dataset.upper()}_{backbone_name}_{img_size}.pth"

    # ================= CAA initialization =================
    class_aware_aug = ClassAwareAugmentation(
        config         = config,
        cache_name     = cache_name,
        img_transform  = img_transform,
        mask_transform = mask_transform
    )

    # ================= Step 1: Build the CAA per-category path dictionary (executed only once globally) =================
    normal_dict, anomaly_img_dict, anomaly_mask_dict = build_caa_category_dict(config=config)
    class_aware_aug.set_category_samples(normal_dict, anomaly_img_dict, anomaly_mask_dict)
    print(f"[CAA] Loaded per-category path dictionary for {dataset}"
          f" ({len(normal_dict)} categories)")

    # ================= Step 2: The training set directly reuses the inference dataset loading =================
    train_dataset = get_dataset(
        config         = config,
        img_transform  = img_transform,
        mask_transform = mask_transform,        
    )

    g = torch.Generator()
    g.manual_seed(config["seed_value"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    # ================= Model =================
    backbone = BackboneWrapper(config=config).to(device).eval()

    detector = Detector(config=config).to(device).train()

    optimizer = optim.AdamW([
        {"params": [*detector.cls_weights], "lr": lr, "weight_decay": cls_wd},
        {"params": [*detector.seg_weights], "lr": lr, "weight_decay": seg_wd},
        {"params": [detector.cls_fusion_weight], "lr": lr, "weight_decay": 0.0},
        {"params": [detector.seg_fusion_weight], "lr": lr, "weight_decay": 0.0},
        
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    cls_criterion = nn.CrossEntropyLoss()
    focal_criterion = FocalLoss()
    dice_criterion = DiceLoss()    

    for epoch in range(epochs):
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for imgs, masks, labels, defect_types, img_names, cats, img_paths in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            # CAA class-aware augmentation
            aug_imgs, aug_masks = [], []
            for i in range(len(imgs)):
                aug_img, aug_mask = class_aware_aug(
                    imgs[i].cpu(), masks[i].cpu(),
                    cats[i], labels[i].item() == 0
                )
                aug_imgs.append(aug_img)
                aug_masks.append(aug_mask)

            imgs = torch.stack(aug_imgs).to(device)
            masks = torch.stack(aug_masks).to(device)

            feats = backbone.extract_features(imgs)
            cls_pred, seg_pred = detector(feats)

            loss_cls = cls_criterion(cls_pred, labels.to(device))

            loss_seg = 0.0
            if seg_pred is not None:
                B, N, C = seg_pred.shape
                S = int(N ** 0.5)
                seg_pred = rearrange(seg_pred, 'b (h w) c -> b c h w', h=S, w=S)
                seg_pred = F.interpolate(
                    seg_pred,
                    size=masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                loss_seg = focal_criterion(seg_pred, masks.long()) + dice_criterion(seg_pred, masks.long())

            optimizer.zero_grad()

            total_loss_step = loss_cls + loss_seg
            total_loss_step.backward()

            optimizer.step()

            batch_loss = loss_cls.item() + loss_seg.item()
            total_loss += batch_loss

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} training complete | average loss: {avg_loss:.4f}")

        scheduler.step()
        
        if ckpt_save_epoch > epochs:
            ckpt_save_epoch = epochs            
        
        if epoch + 1 >= ckpt_save_epoch:
            os.makedirs(checkpoint_dir, exist_ok=True)

            save_path = os.path.join(
                checkpoint_dir, 
                f"backbone_{backbone_name}_size_{img_size}_dataset_{dataset}_fusion_{logits_fusion}_epoch_{epoch + 1}.pth"
            )
        
            torch.save({
                "detector": detector.state_dict(),
                "config": config,
            }, save_path)
        
            print("Model saved to: ", save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LUMIN unified training script, supporting CLIP/DINO multi-backbone and multi-dataset batch experiments")

    # Hardware and random seed
    parser.add_argument("--cuda", type=str, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, help="global random seed")

    # Backbone configuration
    parser.add_argument("--backbone_name", type=str, choices=["CLIP", "DINO"])
    parser.add_argument("--clip_name", type=str)
    parser.add_argument("--dino_path", type=str)
    
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--feat_layers", nargs="+", type=int)
    
    # Dataset
    parser.add_argument("--dataset", type=str, choices=["mvtec","visa","btad","ksdd","real-iad"])
    parser.add_argument("--split", type=str)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--ckpt_save_epoch", type=int)
    
    parser.add_argument("--aug_prob", type=float)
    
    parser.add_argument("--cls_wd", type=float)
    parser.add_argument("--seg_wd", type=float)
    
    # Cache/output paths
    parser.add_argument("--caa_cache_dir", type=str)
    parser.add_argument("--checkpoint_dir", type=str)

    args = parser.parse_args()    
    
    overwrite_map = {
        "cuda": "cuda",
        "seed": "seed_value",
        "backbone_name": "backbone",
        "clip_name": "clip_backbone",
        "dino_path": "dino_backbone",
        "embed_dim": "embed_dim",
        "feat_layers": "feat_layers",
        "dataset": "dataset",
        "split": "split",
        "batch_size": "batch_size",
        "lr": "lr",
        "epochs": "epochs",
        "aug_prob": "aug_prob",
        "cls_wd": "cls_wd",
        "seg_wd": "seg_wd",
        "caa_cache_dir": "caa_cache_dir",
        "checkpoint_dir": "checkpoint_dir"
    }

    for arg_key, cfg_key in overwrite_map.items():
        arg_val = getattr(args, arg_key)
        if arg_val is not None:

            # Override the config dictionary
            config[cfg_key] = arg_val
            
    cuda_id = config["cuda"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    
    global device
    device = torch.device("cuda")
    
    train(config=config)

