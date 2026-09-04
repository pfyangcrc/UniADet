import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

from skimage.measure import label
import cv2
import os

# ===================== Decoupled detector + Mapping Networks (8-layer feature adaptation) =====================
class Detector(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim   = config["embed_dim"]
        self.feat_layers = config["feat_layers"]
        self.num_layers  = len(self.feat_layers)
        
        self.fusion      = config["logits_fusion"]

        self.cls_weights = nn.ParameterList([
            nn.Parameter(torch.randn(2, self.embed_dim)) for _ in range(self.num_layers)
        ])
        self.seg_weights = nn.ParameterList([
            nn.Parameter(torch.randn(2, self.embed_dim)) for _ in range(self.num_layers)
        ])

        # Learnable multi-layer feature fusion weights
        self.cls_fusion_weight = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        self.seg_fusion_weight = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)

        self._init_weights()

    def _init_weights(self):
        for w in self.cls_weights:
            nn.init.xavier_uniform_(w)
        for w in self.seg_weights:
            nn.init.xavier_uniform_(w)

    def forward(self, features):
        cls_preds = []
        seg_preds = []

        scale = 1 / (self.embed_dim ** 0.5)

        for i, feat in enumerate(features[:self.num_layers]):
            cls_feat = feat[:, 0, :]  # [B, D]
            seg_feat = feat[:, 1:, :]  # [B, N, D]
            w_cls = self.cls_weights[i]  # [2, D]
            w_seg = self.seg_weights[i]  # [2, D]

            # ========== Classification branch: normalization + matrix multiplication instead of cosine_similarity ==========
            # Normalize features and weights
            cls_feat_norm = F.normalize(cls_feat, p=2, dim=-1)  # [B, D]
            cls_w_norm = F.normalize(w_cls, p=2, dim=-1)  # [2, D]

            # Inner product: [B,D] @ [D,2] = [B,2]
            cls_logits = (cls_feat_norm @ cls_w_norm.transpose(0, 1)) / scale
            cls_preds.append(cls_logits)

            # ========== Segmentation branch: normalization + batched matrix multiplication ==========
            # Normalize features and weights
            seg_feat_norm = F.normalize(seg_feat, p=2, dim=-1)  # [B, N, D]
            seg_w_norm = F.normalize(w_seg, p=2, dim=-1)  # [2, D]

            # Inner product: [B,N,D] @ [D,2] = [B,N,2]
            seg_logits = (seg_feat_norm @ seg_w_norm.transpose(0, 1)) / scale
            seg_preds.append(seg_logits)

        cls_preds = torch.stack(cls_preds)  # [num_layers, B, 2]
        seg_preds = torch.stack(seg_preds)  # [num_layers, B, N, 2]

        if self.fusion == "softmax_layer_fusion":
            cls_weight = F.softmax(self.cls_fusion_weight, dim=0)
            seg_weight = F.softmax(self.seg_fusion_weight, dim=0)
        
            final_cls = (cls_preds * cls_weight.view(self.num_layers, 1, 1)).sum(0)
            final_seg = (seg_preds * seg_weight.view(self.num_layers, 1, 1, 1)).sum(0)
        else:
            final_cls = cls_preds.mean(0)
            final_seg = seg_preds.mean(0)

        return final_cls, final_seg

class NormalMemoryBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = {}  # key: scale layer, value: [total N_patch, D] concatenation of all images
        self.memory_norm = {}  # pre-normalized memory features, to speed up cosine computation

    def build_memory(self, normal_features):
        """
        Supports multiple normal images, added to the memory bank one by one
        normal_features: list of [1, N, D] (a single image)
        """
        for layer_idx, feat in enumerate(normal_features):
            patch_feat = feat[0, 1:, :].detach()  # drop the CLS, store only patches

            if layer_idx not in self.memory:
                self.memory[layer_idx] = patch_feat
            else:
                self.memory[layer_idx] = torch.cat([self.memory[layer_idx], patch_feat], dim=0)

            # Pre-normalize features; subsequent matrix operations are equivalent to cosine similarity
            mem = self.memory[layer_idx]
            self.memory_norm[layer_idx] = torch.nn.functional.normalize(mem, dim=-1)

    def compute_few_shot_score(self, query_features):
        multi_scale_maps = []

        for layer_idx, query_feat in enumerate(query_features):
            if layer_idx not in self.memory:
                continue

            # Query features [B, N, D], memory features [K, D]
            query = query_feat[:, 1:, :]
            memory_norm = self.memory_norm[layer_idx]  # [K, D]
            B, N, D = query.shape

            max_sim = torch.zeros(B, N, device=query.device)

            # Normalize query features
            q_norm = torch.nn.functional.normalize(query, dim=-1)

            # Matrix multiplication = full cosine similarity [B, N, K]
            cos_sim = q_norm @ memory_norm.T

            # Take the maximum similarity
            max_sim = cos_sim.max(dim=-1)[0]

            anomaly_map = 1.0 - max_sim    # global nearest neighbor [B, N]
            multi_scale_maps.append(anomaly_map)

        # Multi-scale fusion
        fused_map = torch.stack(multi_scale_maps).mean(0)
        return fused_map  # [B, N]

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = 1 - (2. * intersection + 1e-6) / (union + 1e-6)

        return dice.mean()


class ClassAwareAugmentation:
    def __init__(
        self, 
        config: dict,
        cache_name=None,
        img_transform=None,
        mask_transform=None,        
    ):

        self.img_size   = config["img_size"]
        self.grid_sizes = config["grid_sizes"]
        self.prob       = config["aug_prob"]

        self.cache_dir  = config["caa_cache_dir"]  # local cache directory
        self.cache_name = cache_name
        os.makedirs(self.cache_dir, exist_ok=True)

        self.img_transform=img_transform
        self.mask_transform = mask_transform

    def set_category_samples(self, normal_samples, anomaly_samples, anomaly_masks):
        self.cat_normal = {}
        self.cat_anomaly = {}
        self.cat_mask = {}

        # ===================== Cache file name =====================
        cache_file = os.path.join(self.cache_dir, self.cache_name)
        if os.path.exists(cache_file):
            # ===================== Load from local =====================
            print("Found CAA cache, loading local files directly...")
            cached_data = torch.load(cache_file, weights_only=False)
            self.cat_normal = cached_data["normal"]
            self.cat_anomaly = cached_data["anomaly"]
            self.cat_mask = cached_data["mask"]
            print("CAA cache loaded successfully!")
            return

        # ===================== No cache, normal preprocessing =====================
        print("CAA cache not found, starting preprocessing and saving to local...")
        for cat, paths in tqdm(normal_samples.items(), desc="Loading CAA normal samples"):
            self.cat_normal[cat] = [self.img_transform(Image.open(p).convert("RGB")) for p in paths]

        for cat, paths in tqdm(anomaly_samples.items(), desc="Loading CAA anomaly samples"):
            self.cat_anomaly[cat] = [self.img_transform(Image.open(p).convert("RGB")) for p in paths]

            self.cat_mask[cat] = []
            for p in anomaly_masks[cat]:
                mask = Image.open(p).convert("L")
                mask = self.mask_transform(mask)
                self.cat_mask[cat].append(torch.tensor(np.array(mask) > 0).float())

        # ===================== Save to local =====================
        torch.save({
            "normal": self.cat_normal,
            "anomaly": self.cat_anomaly,
            "mask": self.cat_mask
        }, cache_file)
        print(f"CAA cached locally: {cache_file}")

    def sample_pair(self, category, is_normal):
        if is_normal:
            img = random.choice(self.cat_normal[category])
            mask = torch.zeros(self.img_size, self.img_size)
        else:
            if random.random() > 0.5:
                img = random.choice(self.cat_normal[category])
                mask = torch.zeros(self.img_size, self.img_size)
            else:
                idx = random.randint(0, len(self.cat_anomaly[category]) - 1)
                img = self.cat_anomaly[category][idx]
                mask = self.cat_mask[category][idx]
        return img, mask

    def grid_mosaic(self, category, is_normal, grid_size):
        h = w = self.img_size
        mosaic_img = torch.zeros(3, h * grid_size, w * grid_size)
        mosaic_mask = torch.zeros(h * grid_size, w * grid_size)

        for i in range(grid_size):
            for j in range(grid_size):
                img_p, mask_p = self.sample_pair(category, is_normal)
                mosaic_img[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = img_p
                mosaic_mask[i * h:(i + 1) * h, j * w:(j + 1) * w] = mask_p

        mosaic_img = F.interpolate(mosaic_img[None], (h, w), mode='bilinear', align_corners=False)[0]
        mosaic_mask = F.interpolate(mosaic_mask[None, None], (h, w), mode='bilinear')[0, 0]
        return mosaic_img, mosaic_mask

    def grid_crop(self, img, mask, grid_size, is_normal):
        h, w = self.img_size, self.img_size
        ph, pw = h // grid_size, w // grid_size
        valid = []
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * ph, (i + 1) * ph
                x1, x2 = j * pw, (j + 1) * pw
                if is_normal or mask[y1:y2, x1:x2].sum() > 0:
                    valid.append((y1, y2, x1, x2))

        # If there is no valid region, return the original image directly.
        # Avoids errors caused by center cropping in image preprocessing cropping out defective regions
        if len(valid) == 0:
            return img, mask

        y1, y2, x1, x2 = random.choice(valid)
        crop_img = img[:, y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        crop_img = F.interpolate(crop_img[None], (h, w), mode='bilinear', align_corners=False)[0]
        crop_mask = F.interpolate(crop_mask[None, None], (h, w), mode='bilinear')[0, 0]
        return crop_img, crop_mask

    def __call__(self, img, mask, category, is_normal):
        if random.random() > self.prob:
            return img, mask
        grid_size = random.choice(self.grid_sizes)
        if random.random() > 0.5:
            return self.grid_mosaic(category, is_normal, grid_size)
        else:
            return self.grid_crop(img, mask, grid_size, is_normal)


def visualize_anomaly(img, anomaly_map, save_path, mask, threshold):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]

    anomaly_map = cv2.GaussianBlur(anomaly_map, (3, 3), 0)  # Gaussian blur, smooth small fluctuations

    # ====================== Generate predicted mask ======================
    pred_mask = (anomaly_map >= threshold).astype(np.uint8)

    # ====================== Remove small noise ======================
    #    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)  # morphological opening: remove small white dots
    #    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel) # morphological closing: fill small holes

    # Find contours
    contours_pred, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pred = [c for c in contours_pred if cv2.contourArea(c) > 20]
    cv2.drawContours(img, contours_pred, -1, (0, 0, 255), 2)

    # Draw GT (unchanged)
    if mask is not None:
        mask = mask.astype(np.uint8)
        contours_gt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours_gt, -1, (0, 255, 0), 2)

    cv2.imwrite(save_path, img)


def compute_au_pro(anomaly_maps, gt_masks, max_fpr=0.3, num_thresh=200):
    all_scores = []
    all_masks = []
    for am, gt in zip(anomaly_maps, gt_masks):
        all_scores.append(am.ravel())
        all_masks.append(gt.ravel())

    all_scores = np.concatenate(all_scores)
    all_masks = np.concatenate(all_masks)

    thresholds = np.quantile(all_scores, np.linspace(0, 1, num_thresh))[::-1]

    pros = []
    fprs = []

    for t in thresholds:
        pred = (all_scores >= t).astype(np.uint8)

        gt_inst = label(all_masks)
        inst_ids = np.unique(gt_inst)
        inst_ids = inst_ids[inst_ids != 0]

        if len(inst_ids) == 0:
            pro = 1.0
        else:
            hit = 0
            for iid in inst_ids:
                if np.any(pred[gt_inst == iid]):
                    hit += 1
            pro = hit / len(inst_ids)

        fp = np.sum((pred == 1) & (all_masks == 0))
        tn = np.sum((pred == 0) & (all_masks == 0))
        fpr = fp / (fp + tn + 1e-8)

        pros.append(pro)
        fprs.append(fpr)

    sorted_indices = np.argsort(fprs)
    fprs = np.array(fprs)[sorted_indices]
    pros = np.array(pros)[sorted_indices]

    mask = fprs <= max_fpr
    fprs_masked = fprs[mask]
    pros_masked = pros[mask]

    if len(fprs_masked) < 2:
        return 0.0

    au_pro = np.trapz(pros_masked, fprs_masked) / max_fpr
    return au_pro


def resize_keep_ratio_short_side(input, img_size=512, interpolation=None):
    w, h = input.size
    scale = min(h, w) / img_size
    new_w = int(w / scale)
    new_h = int(h / scale)
    output = input.resize((new_w, new_h), interpolation)
    return output