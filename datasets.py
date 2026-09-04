
"""
==============================================================================
Unified multi-industrial anomaly detection dataset wrapper — dataset.py
==============================================================================
Supported datasets: MVTec AD / VisA / BTAD / KSDD / Real-IAD
Unified output interface: (img_tensor, mask_tensor, img_label, defect_type, img_name, category)
Single external entry point: get_dataset(dataset_name, split, img_transform, mask_transform, config)

Author: LUMIN Project
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List


# ==============================================================================
# 1. MVTec AD dataset
# ==============================================================================
# Dataset source: MVTec Anomaly Detection (MVTec AD)
# Directory structure:
#   {data_root}/
#   └── {category}/
#       ├── train/
#       │   └── good/          # normal training samples
#       ├── test/
#       │   ├── good/          # normal test samples
#       │   ├── broken_large/  # anomaly test samples (grouped by defect type into folders)
#       │   ├── .../
#       │   └── {defect_type}/
#       └── ground_truth/
#           └── {defect_type}/ # pixel-level defect masks (only anomaly samples have them)
#               └── {name}_mask.png
# ==============================================================================

class MVTecDataset(Dataset):
    """MVTec AD dataset wrapper, supporting both train/test flows.

    split="train": only reads normal samples under {cat}/train/good/
    split="test" : reads all samples under {cat}/test/ (normal + anomaly), matching ground_truth masks
    """

    def __init__(self, data_root, split, img_transform, mask_transform, categories="all"):
        """
        Args:
            data_root: MVTec raw data root directory (e.g. /data1/pfyang/MVTec/raw)
            split: "train" | "test"
            img_transform: image preprocessing pipeline
            mask_transform: mask preprocessing pipeline
            categories: optional, the list of categories to load; None means load all categories
        """
        self.data_root = data_root
        self.split = split
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # Discover all categories
        all_categories = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ])
        if categories == "all":
            self.categories = all_categories
        else:
            self.categories = [c for c in categories if c in all_categories]

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for cat in self.categories:
            cat_dir = os.path.join(self.data_root, cat)

            if self.split == "train":
                # Training set: only read normal samples
                good_dir = os.path.join(cat_dir, "train", "good")
                if not os.path.isdir(good_dir):
                    continue
                for img_name in sorted(os.listdir(good_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue
                    img_path = os.path.join(good_dir, img_name)
                    samples.append({
                        "img_path": img_path,
                        "mask_path": None,
                        "label": 0,
                        "defect_type": "good",
                        "img_name": img_name,
                        "category": cat,
                    })
            else:
                # Test set: read all samples (normal + anomaly)
                test_dir = os.path.join(cat_dir, "test")
                gt_dir = os.path.join(cat_dir, "ground_truth")
                if not os.path.isdir(test_dir):
                    continue

                for defect_type in sorted(os.listdir(test_dir)):
                    sub_dir = os.path.join(test_dir, defect_type)
                    if not os.path.isdir(sub_dir):
                        continue

                    for img_name in sorted(os.listdir(sub_dir)):
                        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            continue
                        img_path = os.path.join(sub_dir, img_name)
                        label = 0 if defect_type == "good" else 1

                        # Construct the mask path (only anomaly samples have masks)
                        mask_path = None
                        if defect_type != "good":
                            mask_name = os.path.splitext(img_name)[0] + "_mask.png"
                            candidate = os.path.join(gt_dir, defect_type, mask_name)
                            if os.path.exists(candidate):
                                mask_path = candidate

                        samples.append({
                            "img_path": img_path,
                            "mask_path": mask_path,
                            "label": label,
                            "defect_type": defect_type,
                            "img_name": img_name,
                            "category": cat,
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        mask_path = sample["mask_path"]

        # --- Image loading ---
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.img_transform(img)

        # --- Mask loading ---
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask_tensor = torch.tensor(np.array(mask) > 128, dtype=torch.float32)
        else:
            # No defect / no mask: return an all-zero tensor with the same spatial size as the image
            mask_tensor = torch.zeros(img_tensor.shape[1], img_tensor.shape[2])

        # Validate the cropped mask: CenterCrop may crop out the entire defect region, making the mask all zeros;
        # in that case, force-correct the label to normal to avoid inconsistency between the normal/anomaly label and the mask
        if mask_tensor.max().item() < 1e-6:
            fixed_label = 0
        else:
            fixed_label = sample["label"]

        return (
            img_tensor,
            mask_tensor,
            fixed_label,
            sample["defect_type"],
            sample["img_name"],
            sample["category"],
            sample["img_path"],
        )


# ==============================================================================
# 2. VisA dataset
# ==============================================================================
# Dataset source: Visual Anomaly (VisA)
# Directory structure: CSV-driven path indexing
#   {data_root}/
#   ├── split_csv/
#   │   ├── 2cls_fewshot.csv    # train/test split file
#   │   └── ...
#   ├── {object}/
#   │   ├── Data/Images/Normal/ # normal samples
#   │   ├── Data/Images/{defect}/ # anomaly samples
#   │   └── Data/Masks/{defect}/  # anomaly masks
#   └── ...
# CSV columns: object, image, mask, label, split
# ==============================================================================

class VisADataset(Dataset):
    """VisA dataset wrapper, driven by the CSV split file, supporting both train/test flows.

    split="train": only reads normal samples where split=="train"
    split="test" : reads all samples where split=="test" (normal + anomaly), matching mask paths
    """

    def __init__(self, data_root, split, img_transform, mask_transform,
                 split_file="2cls_fewshot.csv", categories="all"):
        """
        Args:
            data_root: VisA raw data root directory (e.g. /data1/pfyang/VisA/raw)
            split: "train" | "test"
            img_transform: image preprocessing pipeline
            mask_transform: mask preprocessing pipeline
            split_file: the official VisA CSV split filename (located under VisA/raw/split_csv/)
            categories: optional, the list of objects to load; None means load all objects
        """
        self.data_root = data_root
        self.split = split
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.split_file = split_file

        self.samples = self._load_samples()

        # Extract the category list from the loaded samples (deduplicated, order preserved)
        seen = set()
        self.categories = []
        for s in self.samples:
            cat = s["category"]
            if cat not in seen:
                seen.add(cat)
                self.categories.append(cat)

        if categories not in ("all", None):
            self.categories = [c for c in categories if c in self.categories]
            self.samples = [s for s in self.samples if s["category"] in set(categories)]

    def _load_samples(self):
        split_csv = os.path.join(self.data_root, "split_csv", self.split_file)
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"VisA split file does not exist: {split_csv}")

        df = pd.read_csv(split_csv)
        df_split = df[df["split"] == self.split]

        samples = []
        for _, row in df_split.iterrows():
            img_rel = row["image"]
            img_path = os.path.join(self.data_root, img_rel)

            # Mask path (the mask column in the CSV may be empty/NULL)
            mask_rel = row.get("mask", None)
            if pd.notna(mask_rel) and mask_rel is not None and str(mask_rel).strip() != "":
                mask_path = os.path.join(self.data_root, mask_rel)
                if not os.path.exists(mask_path):
                    mask_path = None  # tolerate missing files
            else:
                mask_path = None

            defect_type = str(row["label"])  # "normal" or anomaly type name
            label = 0 if defect_type == "normal" else 1

            samples.append({
                "img_path": img_path,
                "mask_path": mask_path,
                "label": label,
                "defect_type": defect_type,
                "img_name": os.path.basename(img_rel),
                "category": row["object"],
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        mask_path = sample["mask_path"]

        # --- Image loading ---
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.img_transform(img)

        # --- Mask loading ---
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            # VisA original logic uses > 0
            mask_tensor = torch.tensor(np.array(mask) > 0, dtype=torch.float32)
        else:
            mask_tensor = torch.zeros(img_tensor.shape[1], img_tensor.shape[2])

        # Validate the cropped mask: CenterCrop may crop out the entire defect region, making the mask all zeros;
        # in that case, force-correct the label to normal to avoid inconsistency between the normal/anomaly label and the mask
        if mask_tensor.max().item() < 1e-6:
            fixed_label = 0
        else:
            fixed_label = sample["label"]

        return (
            img_tensor,
            mask_tensor,
            fixed_label,
            sample["defect_type"],
            sample["img_name"],
            sample["category"],
            sample["img_path"],
        )


# ==============================================================================
# 3. BTAD dataset
# ==============================================================================
# Dataset source: BeanTech Anomaly Detection (BTAD)
# Directory structure:
#   {data_root}/
#   └── {category}/             # e.g. 01, 02, 03
#       ├── train/
#       │   └── ok/             # normal training samples
#       ├── test/
#       │   ├── ok/             # normal test samples
#       │   └── ko/             # anomaly test samples
#       └── ground_truth/
#           └── ko/             # anomaly pixel-level masks
#               └── {name}.png  # note: BTAD masks do not carry the _mask suffix
# ==============================================================================

class BTADDataset(Dataset):
    """BTAD dataset wrapper, supporting both train/test flows.

    split="train": only reads normal samples under {cat}/train/ok/
    split="test" : reads all samples under {cat}/test/ (ok + ko), matching ground_truth masks
    """

    def __init__(self, data_root, split, img_transform, mask_transform, categories="all"):
        """
        Args:
            data_root: BTAD raw data root directory (e.g. /data1/pfyang/BTAD)
            split: "train" | "test"
            img_transform: image preprocessing pipeline
            mask_transform: mask preprocessing pipeline
            categories: optional, the list of categories to load; None means load all categories
        """
        self.data_root = data_root
        self.split = split
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        all_categories = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ])
        if categories == "all":
            self.categories = all_categories
        else:
            self.categories = [c for c in categories if c in all_categories]

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for cat in self.categories:
            cat_dir = os.path.join(self.data_root, cat)

            if self.split == "train":
                # Training set: only normal samples
                ok_dir = os.path.join(cat_dir, "train", "ok")
                if not os.path.isdir(ok_dir):
                    continue
                for img_name in sorted(os.listdir(ok_dir)):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue
                    img_path = os.path.join(ok_dir, img_name)
                    samples.append({
                        "img_path": img_path,
                        "mask_path": None,
                        "label": 0,
                        "defect_type": "ok",
                        "img_name": img_name,
                        "category": cat,
                    })
            else:
                # Test set: normal + anomaly
                test_dir = os.path.join(cat_dir, "test")
                gt_dir = os.path.join(cat_dir, "ground_truth")
                if not os.path.isdir(test_dir):
                    continue

                for defect_type in sorted(os.listdir(test_dir)):
                    sub_dir = os.path.join(test_dir, defect_type)
                    if not os.path.isdir(sub_dir):
                        continue

                    for img_name in sorted(os.listdir(sub_dir)):
                        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            continue
                        img_path = os.path.join(sub_dir, img_name)
                        label = 0 if defect_type == "ok" else 1

                        # BTAD masks: all masks are under ground_truth/ko, matched by basename ignoring the extension
                        mask_path = None
                        if defect_type == "ko":
                            base = os.path.splitext(img_name)[0]
                            ko_dir = os.path.join(gt_dir, "ko")
                            if os.path.isdir(ko_dir):
                                for f in os.listdir(ko_dir):
                                    if os.path.splitext(f)[0] == base:
                                        mask_path = os.path.join(ko_dir, f)
                                        break

                        samples.append({
                            "img_path": img_path,
                            "mask_path": mask_path,
                            "label": label,
                            "defect_type": defect_type,
                            "img_name": img_name,
                            "category": cat,
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        mask_path = sample["mask_path"]

        # --- Image loading ---
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.img_transform(img)

        # --- Mask loading ---
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask_tensor = torch.tensor(np.array(mask) > 128, dtype=torch.float32)
        else:
            mask_tensor = torch.zeros(img_tensor.shape[1], img_tensor.shape[2])

        # Validate the cropped mask: CenterCrop may crop out the entire defect region, making the mask all zeros;
        # in that case, force-correct the label to normal to avoid inconsistency between the normal/anomaly label and the mask
        if mask_tensor.max().item() < 1e-6:
            fixed_label = 0
        else:
            fixed_label = sample["label"]

        return (
            img_tensor,
            mask_tensor,
            fixed_label,
            sample["defect_type"],
            sample["img_name"],
            sample["category"],
            sample["img_path"],
        )


# ==============================================================================
# 4. KSDD dataset
# ==============================================================================
# Dataset source: Kolektor Surface-Defect Dataset (KSDD)
# Directory structure (images and masks are in the same subfolder):
#   {data_root}/
#   └── KolektorSDD/
#       ├── kos01/                 # memory bank normal-sample range (kos01 ~ kos10)
#       │   ├── Part0.jpg          # raw image
#       │   ├── Part0_label.bmp    # corresponding pixel-level defect mask
#       │   └── ...
#       ├── ...
#       ├── kos10/
#       └── kos11/                 # test-set range (kos11 ~ kos50)
#           └── ...
#
# Data split rules:
#   - split="train": only loads normal samples with all-zero masks in kos01~kos10 (for the memory bank, 70 images)
#   - split="test" : loads all samples in kos11~kos50 (normal + defective, 319 images), with masks
# ==============================================================================

def _is_mask_all_zero(mask_path):
    """Determine whether the bmp mask is all black with no defective pixels."""
    mask_arr = np.array(Image.open(mask_path))
    return mask_arr.max() == 0


def _parse_kos_number(dirname):
    """Extract the number XX from a 'kosXX' directory name; return -1 for non-kos directories."""
    if not dirname.startswith("kos"):
        return -1
    try:
        return int(dirname[3:])
    except ValueError:
        return -1


def _get_ksdd_good_paths(ksdd_root):
    """Traverse KolektorSDD/kos01~kos10 and collect paths of normal samples with all-zero masks.

    Used directly by build_memory in infer.py to avoid depending on a non-existent train/good directory.
    """
    img_dir = os.path.join(ksdd_root, "KolektorSDD")
    good_paths = []
    if not os.path.isdir(img_dir):
        return good_paths
    for dirname in sorted(os.listdir(img_dir)):
        kos_num = _parse_kos_number(dirname)
        if kos_num < 1 or kos_num > 10:
            continue  # kos01~kos10 range
        sdir = os.path.join(img_dir, dirname)
        if not os.path.isdir(sdir):
            continue
        for fname in sorted(os.listdir(sdir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if '_label' in fname.lower():
                continue
            base = os.path.splitext(fname)[0]
            mask_path = os.path.join(sdir, base + "_label.bmp")
            if os.path.exists(mask_path) and _is_mask_all_zero(mask_path):
                good_paths.append(os.path.join(sdir, fname))
    return good_paths
    

class KSDDDataset(Dataset):
    """KSDD dataset wrapper, distinguishing train/test by kos ranges.

    split="train": loads kos01~kos10, keeping only all-zero-mask normal samples (for the memory bank)
    split="test" : loads kos11~kos50, all samples (normal + defective), matching the corresponding masks
    """

    # kos range segments
    TRAIN_KOS_START = 1
    TRAIN_KOS_END   = 10
    TEST_KOS_START  = 11
    TEST_KOS_END    = 50

    def __init__(self, data_root, split, img_transform, mask_transform, categories="all"):
        self.data_root = data_root
        self.split = split
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.img_dir = os.path.join(data_root, "KolektorSDD")

        # KSDD has only one category
        self.categories = ["KSDD"]

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for dirname in sorted(os.listdir(self.img_dir)):
            sdir = os.path.join(self.img_dir, dirname)
            if not os.path.isdir(sdir):
                continue

            kos_num = _parse_kos_number(dirname)
            if kos_num < 0:
                continue  # skip non-kos directories

            # ---- Segment by kos number ----
            if self.split == "train":
                # Memory bank range kos01~kos10, only keep normal samples with all-zero masks
                if kos_num < self.TRAIN_KOS_START or kos_num > self.TRAIN_KOS_END:
                    continue
            else:
                # Test range kos11~kos50, all samples
                if kos_num < self.TEST_KOS_START or kos_num > self.TEST_KOS_END:
                    continue

            for img_name in sorted(os.listdir(sdir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                if '_label' in img_name.lower():
                    continue  # skip the mask file itself

                img_path = os.path.join(sdir, img_name)

                # Match masks in the same directory: PartN.jpg → PartN_label.bmp
                base = os.path.splitext(img_name)[0]
                mask_name = base + "_label.bmp"
                mask_path = os.path.join(sdir, mask_name)

                if os.path.exists(mask_path):
                    if _is_mask_all_zero(mask_path):
                        # all-zero mask → normal sample
                        mask_path_final = None
                        label = 0
                        defect_type = "good"
                    else:
                        # mask contains defective pixels
                        mask_path_final = mask_path
                        label = 1
                        defect_type = "defect"
                else:
                    # no mask file → normal sample
                    mask_path_final = None
                    label = 0
                    defect_type = "good"

                # train mode keeps only normal samples
                if self.split == "train" and label == 1:
                    continue

                samples.append({
                    "img_path": img_path,
                    "mask_path": mask_path_final,
                    "label": label,
                    "defect_type": defect_type,
                    "img_name": f"{dirname}/{img_name}",
                    "category": "KSDD",
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        mask_path = sample["mask_path"]

        # --- Image loading ---
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.img_transform(img)

        # --- Mask loading ---
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask_tensor = torch.tensor(np.array(mask) > 128, dtype=torch.float32)
        else:
            mask_tensor = torch.zeros(img_tensor.shape[1], img_tensor.shape[2])

        # Validate the cropped mask: CenterCrop may crop out the entire defect region, making the mask all zeros;
        # in that case, force-correct the label to normal to avoid inconsistency between the normal/anomaly label and the mask
        if mask_tensor.max().item() < 1e-6:
            fixed_label = 0
        else:
            fixed_label = sample["label"]

        return (
            img_tensor,
            mask_tensor,
            fixed_label,
            sample["defect_type"],
            sample["img_name"],
            sample["category"],
            sample["img_path"],
        )


# ==============================================================================
# 5. Real-IAD dataset
# ==============================================================================
# Dataset source: Real-Industrial Anomaly Detection (Real-IAD)
# Directory structure:
#   {data_root}/
#   └── {category}/               # e.g. audiojack, phone, ...
#       ├── train/
#       │   └── good/             # normal training samples (multi-view scene subdirectories)
#       ├── OK/                   # normal test samples (all without masks)
#       │   └── {scene}/
#       │       └── {name}.jpg
#       └── NG/                   # anomaly test samples
#           └── {defect_type}/
#               └── {scene}/
#                   ├── {name}.jpg   # RGB raw image
#                   └── {name}.png   # pixel-level defect mask (same name prefix, .png suffix)
#
# Note: inside NG folders there are two separate files in the same directory,
#       xxx.jpg (raw image) + xxx.png (mask), matched exactly by the filename stem;
#       samples without a same-name png mask are treated as normal.
# ==============================================================================

class RealIADDataset(Dataset):
    """Real-IAD dataset wrapper, supporting both train/test flows.

    split="train": only reads normal samples under {cat}/OK/ (for memory bank construction)
    split="test" : only reads all samples under {cat}/NG/ (normal + anomaly),
                  matching same-name .png masks in the same directory; no .png → normal sample
    """

    def __init__(self, data_root, split, img_transform, mask_transform, categories="all"):
        self.data_root = data_root
        self.split = split
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        all_categories = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
            # Filter out non-data directories (e.g. realiad_jsons)
            and not d.endswith('_jsons')
        ])
        if categories == "all":
            self.categories = all_categories
        else:
            self.categories = [c for c in categories if c in all_categories]

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for cat in self.categories:
            cat_dir = os.path.join(self.data_root, cat)

            if self.split == "train":
                # ---- Training set: OK directory (all normal, used for the memory bank) ----
                ok_dir = os.path.join(cat_dir, "OK")
                if not os.path.isdir(ok_dir):
                    continue
                for scene in sorted(os.listdir(ok_dir)):
                    sd = os.path.join(ok_dir, scene)
                    if not os.path.isdir(sd):
                        continue
                    for img_name in sorted(os.listdir(sd)):
                        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            continue
                        img_path = os.path.join(sd, img_name)
                        samples.append({
                            "img_path": img_path, "mask_path": None, "label": 0,
                            "defect_type": "good", "img_name": img_name, "category": cat,
                        })
            else:
                # ---- Test set: only the NG directory (jpg raw image + same-name png mask matching) ----
                ng_dir = os.path.join(cat_dir, "NG")
                if not os.path.isdir(ng_dir):
                    continue
                for defect_type in sorted(os.listdir(ng_dir)):
                    dd = os.path.join(ng_dir, defect_type)
                    if not os.path.isdir(dd):
                        continue
                    for scene in sorted(os.listdir(dd)):
                        sd = os.path.join(dd, scene)
                        if not os.path.isdir(sd):
                            continue
                        for img_name in sorted(os.listdir(sd)):
                            if not img_name.lower().endswith(('.jpg', '.jpeg')):
                                continue
                            img_path = os.path.join(sd, img_name)
                            base = os.path.splitext(img_name)[0]
                            mask_path = os.path.join(sd, base + ".png")
                            if os.path.exists(mask_path):
                                label, defect = 1, defect_type
                            else:
                                mask_path, label, defect = None, 0, "OK"
                            samples.append({
                                "img_path": img_path, "mask_path": mask_path, "label": label,
                                "defect_type": defect, "img_name": img_name, "category": cat,
                            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        img_tensor = self.img_transform(img)
        mask_path = sample["mask_path"]
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask_tensor = torch.tensor(np.array(mask) > 128, dtype=torch.float32)
        else:
            mask_tensor = torch.zeros(img_tensor.shape[1], img_tensor.shape[2])
        if mask_tensor.max().item() < 1e-6:
            fixed_label = 0
        else:
            fixed_label = sample["label"]
        return (img_tensor, mask_tensor, fixed_label,
                sample["defect_type"], sample["img_name"], sample["category"])
# ==============================================================================
# 6. Top-level factory function
# ==============================================================================

# dataset name → (class, required config key → constructor argument)
_DATASET_REGISTRY = {
    "mvtec": {
        "class": MVTecDataset,
        "root_key": "mvtec_data_root",
        "extra_kwargs": [],
    },
    "visa": {
        "class": VisADataset,
        "root_key": "visa_data_root",
        "extra_kwargs": ["visa_split_file"],
    },
    "btad": {
        "class": BTADDataset,
        "root_key": "btad_data_root",
        "extra_kwargs": [],
    },
    "ksdd": {
        "class": KSDDDataset,
        "root_key": "ksdd_data_root",
        "extra_kwargs": [],
    },
    "real-iad": {
        "class": RealIADDataset,
        "root_key": "realiad_data_root",
        "extra_kwargs": [],
    },
}


def get_dataset(config, img_transform, mask_transform, categories="all"):
    """Top-level factory function — the single external entry point for training/inference.

    Args:
        config: dictionary, must contain the root path field for the corresponding dataset, e.g.:
        {
            "visa_data_root": "/data1/pfyang/VisA/raw",
            "visa_split_file": "2cls_fewshot.csv",

            "mvtec_data_root": "/data1/pfyang/MVTec/raw",
            "btad_data_root": "/data1/pfyang/BTAD",
            "ksdd_data_root": "/data1/pfyang/KSDD",
            "realiad_data_root": "/data1/pfyang/Real-IAD/Real-IAD/Real-IAD",
        }
        img_transform: image preprocessing pipeline (e.g. transforms.Compose([...]))
        mask_transform: mask preprocessing pipeline

        categories: optional, the list of categories to load; None means load all

    Returns:
        the corresponding dataset instance (a torch.utils.data.Dataset subclass)

    Raises:
        ValueError: invalid dataset_name
        FileNotFoundError: data directory / split file does not exist

    dataset_name: dataset name, allowed values:
                  "mvtec"   — MVTec AD
                  "visa"    — VisA
                  "btad"    — BTAD
                  "ksdd"    — KSDD
                  "real-iad"— Real-IAD
    split       : "train" | "test" (training usually uses "test" to load all normal + anomaly samples)
    """
    dataset_name = config["dataset"]
    split        = config["split"]    
    
    dataset_name = dataset_name.lower().strip()

    if dataset_name not in _DATASET_REGISTRY:
        valid = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported dataset name '{dataset_name}', "
            f"allowed values: {valid}"
        )

    entry = _DATASET_REGISTRY[dataset_name]
    dataset_cls = entry["class"]
    root_key = entry["root_key"]

    # Read the data root directory
    data_root = config.get(root_key)
    if data_root is None or not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"[{dataset_name}] data root directory does not exist or is not configured: "
            f"config['{root_key}'] = {data_root}"
        )

    # Build the constructor arguments
    init_kwargs = {
        "data_root": data_root,
        "split": split,
        "img_transform": img_transform,
        "mask_transform": mask_transform,
    }

    # Dataset-specific parameters
    for kw in entry["extra_kwargs"]:
        if kw == "visa_split_file":
            init_kwargs["split_file"] = config.get("visa_split_file", "2cls_fewshot.csv")

    if categories is not None:
        init_kwargs["categories"] = categories

    return dataset_cls(**init_kwargs)


# ==============================================================================
# 7. CAA per-category path dictionary extraction utility
# ==============================================================================
# Designed for the ClassAwareAugmentation pre-built cache: traverses the dataset samples list once,
# groups normal/anomaly image paths and anomaly mask paths by category, without image loading or transforms.
# CAA initialization / local cache is invoked only once and does not depend on a training-specific Dataset class.
# ==============================================================================

def build_caa_category_dict(config, categories="all"):
    """Extract the three per-category path dictionaries required by CAA.

    Internally calls get_dataset to obtain the sample list (paths only, no image/mask transforms),
    then returns a triple after grouping, for use by ClassAwareAugmentation.set_category_samples().

    Args:
        config: unified config dictionary (containing the root paths of all datasets)
        categories: optional category subset; None means all

    Returns:
        (normal_images_by_category: Dict[str, List[str]],
         anomaly_images_by_category: Dict[str, List[str]],
         anomaly_mask_by_category: Dict[str, List[str]])
    """
    # Reuse the dataset loading approach (transform set to identity, only paths are taken)
    base_ds = get_dataset(
        config=config,
        img_transform=lambda x: x,       # no image transform, placeholder only
        mask_transform=lambda x: x,      # no mask transform, placeholder only
        categories=categories,
    )

    normal_images_by_category: Dict[str, List[str]] = {}
    anomaly_images_by_category: Dict[str, List[str]] = {}
    anomaly_mask_by_category: Dict[str, List[str]] = {}

    for cat in base_ds.categories:
        normal_images_by_category[cat] = []
        anomaly_images_by_category[cat] = []
        anomaly_mask_by_category[cat] = []

    for sample in base_ds.samples:
        cat = sample["category"]
        img_path = sample["img_path"]
        mask_path = sample.get("mask_path", None)

        if sample["label"] == 0:
            normal_images_by_category[cat].append(img_path)
        else:
            anomaly_images_by_category[cat].append(img_path)
            anomaly_mask_by_category[cat].append(mask_path if mask_path is not None else None)

    return normal_images_by_category, anomaly_images_by_category, anomaly_mask_by_category