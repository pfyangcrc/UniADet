import sys
import os
import yaml
import argparse
import importlib
import gc
import csv
import time
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbones import BackboneWrapper
from modules import (
    Detector,
    NormalMemoryBank,
    visualize_anomaly,
    resize_keep_ratio_short_side as resize_img
)

# Unified dataset interface
from datasets import get_dataset, _get_ksdd_good_paths

# ===================== Sampling method registry =====================
_sample_mod = importlib.import_module("memory.sample_methods")
SAMPLING_METHODS = {
    "random_sampling": _sample_mod.random_sampling,
}

# ==============================================================================
# Load inference config
# ==============================================================================
def load_config(config_path="config_infer.yaml"):
    """Load the inference config from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

# ==============================================================================
# Utility functions (original logic fully preserved, no changes)
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Dataset → normal-sample subdirectory mapping
_GOOD_DIR_RULES = {
    "mvtec":    lambda root, cat: os.path.join(root, cat, "train", "good"),
    "visa":     lambda root, cat: os.path.join(root, cat, "Data", "Images", "Normal"),
    "btad":     lambda root, cat: os.path.join(root, cat, "train", "ok"),
    "ksdd":     lambda root, cat: _get_ksdd_good_paths(root),
    "real-iad": lambda root, cat: os.path.join(root, cat, "OK"),
}

def _get_dataset_config(config):
    """Derive the current dataset-related paths from the passed config, no longer relying on the global config"""
    dataset = config["dataset"]
    root_key_map = {
        "mvtec": "mvtec_data_root",
        "visa": "visa_data_root",
        "btad": "btad_data_root",
        "ksdd": "ksdd_data_root",
        "real-iad": "realiad_data_root",
    }
    data_root = config[root_key_map[dataset]]
    
    # Automatically name output paths
    memory_bank_dir = f"{config['memory_bank_dir']}/{config['backbone']}/{dataset.upper() if dataset != 'real-iad' else 'RealIAD'}"
    csv_path = f"{config['csv_path']}/results_{dataset}.csv"
    vis_results_dir = f"{config['vis_results_dir']}/{dataset.upper() if dataset != 'real-iad' else 'RealIAD'}"
    
    return data_root, memory_bank_dir, csv_path, vis_results_dir

# ===================== Build the memory bank (only adds config as a parameter, removes the global config dependency) =====================
def build_memory(backbone, good_dir, k_shot, img_transform, cat, config):
    start_mem_time = time.time()
    selected_num = 0
    selected_img_list = ""
    timings = {}
    
    sample_method = config["sample_method"]
    sample_layer  = config["sample_layer"]
    
    with torch.no_grad():
        _, memory_bank_dir, _, _ = _get_dataset_config(config)        
        bank_dir  = f"{memory_bank_dir}/{sample_method}_f_{sample_layer}/{k_shot}_shot"
        os.makedirs(bank_dir, exist_ok=True)
        bank_file = os.path.join(bank_dir, f"memory_{cat}.pth")
        
        if os.path.exists(bank_file):
            print(f"\nFound memory bank cache, loading local files directly...")
            bank_data = torch.load(bank_file, map_location=device, weights_only=False)
            memory_bank       = bank_data["memory_bank"]
            selected_num      = bank_data["selected_num"]
            selected_img_list = bank_data["selected_img_list"]

            mem_load_time     = time.time() - start_mem_time
            print(f"\n[OK] Memory bank loaded, time: {mem_load_time:.2f}s\n")

            # Directly retrieve the full timing dictionary
            timings = bank_data.get("timings", {})
            # Override this load time (no load time exists when building from cache)
            timings["mem_load_time"] = round(mem_load_time, 3)

            return memory_bank, selected_num, selected_img_list, timings


        print(f"\nMemory bank cache not found, starting to build and save to local...")
        memory_bank = NormalMemoryBank()
        
        if isinstance(good_dir, list):
            img_paths = good_dir
        elif os.path.isdir(good_dir):
            img_paths = []
            for root, _, files in os.walk(good_dir):
                for f in files:
                    if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                        img_paths.append(os.path.join(root, f))
        else:
            img_paths = []
            
        if not img_paths:
            print(f"  [WARN] good_dir has no images: {good_dir}, skipping memory bank construction")
            return None, 0, "", {}
        
        if len(img_paths) > k_shot:
            print(f"Sampling method: {sample_method} (category={cat})...")

            random.shuffle(img_paths)
            selected_paths = img_paths[:sample_num]
            timings = {}

        selected_num = len(selected_paths)
        selected_img_list = ", ".join([os.path.basename(p) for p in selected_paths])

        for img_path in selected_paths:
            img   = Image.open(img_path).convert("RGB")
            img   = img_transform(img).unsqueeze(0).to(device)
            feats = backbone.extract_features(img)            
            memory_bank.build_memory(feats)
            
        mem_build_time = time.time() - start_mem_time
        meta_time      = timings.get("meta_total_time", 0)
        
        if meta_time > 0:
            mem_build_time = mem_build_time - meta_time
            
        timings["mem_build_time"] = round(mem_build_time, 3)
        
        torch.save({
            "memory_bank"       : memory_bank,
            "selected_num"      : selected_num,
            "selected_img_list" : selected_img_list,
            "timings"           : timings,
        }, bank_file)
        
        print(f"\n[OK] Memory bank built and saved to local: {bank_file}, time: {mem_build_time:.2f}s\n")
        
        timings["mem_load_time"] = 0.0
        timings["mem_build_time"] = round(mem_build_time, 3)
        
    return memory_bank, selected_num, selected_img_list, timings

# ===================== Batch inference function (no global config dependency, only depends on passed parameters) =====================
def infer_batch(backbone, detector, memory_bank, imgs, k_shot, mask_size, few_shot_weight, config):
    with torch.no_grad():
        feats = backbone.extract_features(imgs)
        cls_pred, seg_pred = detector(feats)
        cls_prob = F.softmax(cls_pred, dim=-1)[:, 1]
        
        if seg_pred is not None:
            seg_prob = F.softmax(seg_pred, dim=-1)[..., 1]
            
            if k_shot > 0:
                fsw = few_shot_weight
                few_shot_map = memory_bank.compute_few_shot_score(feats)
                seg_prob = (1 - fsw) * seg_prob + fsw * few_shot_map
                
                flat_fuse = seg_prob.flatten(1)                
                cls_fewshot = flat_fuse.max(dim=-1).values
                cls_prob = (1 - fsw) * cls_prob + fsw * cls_fewshot
                
            B, N = seg_prob.shape
            s = int(N ** 0.5)
            seg_prob = seg_prob.view(-1, 1, s, s)
            seg_prob = F.interpolate(seg_prob, size=mask_size, mode="bilinear", align_corners=False)
            seg_maps = seg_prob.squeeze(1).cpu().numpy()
        else:
            seg_maps = None
            cls_prob = None
            
    return cls_prob, seg_maps

# ===================== Single-category evaluation main function (config and device passed in, global config removed) =====================
def evaluate(backbone, detector, k_shot, few_shot_weight, cat, config):
    device = torch.device("cuda")

    seed_value      = config["seed_value"]    
    set_seed(seed_value)
    
    dataset         = config["dataset"]
    data_root, _, csv_path, _ = _get_dataset_config(config)
    
    batch_size      = config["batch_size"]
    num_workers     = config["num_workers"]
    backbone_name   = config["backbone"]
    
    vis_results_dir = config["vis_results_dir"]
    sample_method   = config["sample_method"]
    sample_layer    = config["sample_layer"]
    model_filename  = config["model_filename"]
    
    verbose         = config["verbose"]
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.max_memory_allocated() / (1024**3)
    
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
        
    # On Windows, it must be written this way to avoid errors
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
    vis_transform = transforms.Compose([
        transforms.Lambda(resize_img_bilinear),
        transforms.CenterCrop(img_size),
    ])
    print(f"\n{'='*55}")
    print(f"Dataset: {dataset} | category: {cat}")
    print(f"model_filename = {model_filename}")
    print(f"k_shot = {k_shot} | few_shot_weight = {few_shot_weight}")
    print(f"sample_method = {sample_method} | sample_layer = {sample_layer}")
    print(f"{'='*55}")
    
    # ---- Get good_dir according to the dataset rule ----
    good_dir_rule = _GOOD_DIR_RULES.get(dataset)
    if good_dir_rule is None:
        raise ValueError(f"Unsupported dataset: {dataset}")
    good_dir = good_dir_rule(data_root, cat)
    
    # ===================== Build the memory bank =====================
    if k_shot > 0:
        memory_bank, sel_num, sel_img_list, timings = build_memory(
            backbone, good_dir, k_shot, img_transform, cat, config
        )
    else:
        memory_bank = None
        sel_num = 0
        sel_img_list = ""
        timings = {}
    # ===================== Create the test set using the unified dataset interface =====================
    test_dataset = get_dataset(
        config=config,
        img_transform=img_transform,
        mask_transform=mask_transform,        
        categories=[cat],
    )
    
    g = torch.Generator()
    g.manual_seed(seed_value)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False
    )
    
    all_img_scores, all_img_labels = [], []
    all_pix_scores, all_pix_masks = [], []
    
    start_cat_time = time.time()
    
    pbar = tqdm(test_loader, desc=f"[{dataset}/{cat}] inferring")
    for batch in pbar:
        imgs, masks, labels, defect_types, img_names, categories, img_paths = batch
        imgs = imgs.to(device)
        
        cls_prob, seg_maps = infer_batch(
            backbone, detector, memory_bank,
            imgs, k_shot, masks.shape[-2:], 
            few_shot_weight, config
        )
        
        all_img_scores.append(cls_prob.detach().cpu().numpy())
        
        for i in range(len(imgs)):
            seg_map = seg_maps[i]
            mask = masks[i].numpy()
            defect_type = defect_types[i]
            img_name = img_names[i]
            all_img_labels.append(labels[i].item())
            all_pix_scores.append(seg_map.flatten())
            all_pix_masks.append(mask.flatten().astype(np.int64))
            # Visualization

            if config["vis"] == 1:     
                ds_upper = dataset.upper() if dataset != "real-iad" else "RealIAD"
                vis_results_path = (
                    f"{vis_results_dir}/{ds_upper}"
                    f"/{sample_method}_f_{sample_layer}/{k_shot}_shot/{model_filename}/{cat}"
                )
                os.makedirs(vis_results_path, exist_ok=True)
                save_path = os.path.join(vis_results_path, f"{defect_type}_{img_name}")
                
                raw_img = Image.open(img_paths[i]).convert("RGB")
                vis_img = vis_transform(raw_img)
                visualize_anomaly(vis_img, seg_map, save_path, mask, threshold=config["vis_threshold"])
                
    # ===================== Metric computation =====================
    cat_time = time.time() - start_cat_time
    img_count = len(test_dataset)
    avg_time = cat_time / img_count if img_count > 0 else 0
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    infer_mem = max(peak_mem - base_mem, 0.0)
    
    if config["metrics"] == 1:
        img_metric_start = time.time()
       
        # ========== Original image code logic ==========
        all_img_labels_np = np.array(all_img_labels)
        all_img_scores_np = np.concatenate(all_img_scores)        
        i_auroc = 100 * roc_auc_score(all_img_labels_np, all_img_scores_np)
        i_aupr  = 100 * average_precision_score(all_img_labels_np, all_img_scores_np)        
        ip, ir, _ = precision_recall_curve(all_img_labels_np, all_img_scores_np)
        i_f1max = 100 * np.max(2 * ip * ir / (ip + ir + 1e-8))
        
        img_metric_cost  = time.time() - img_metric_start
        
        pix_metric_start = time.time()
        
                # Stratified pixel sampling function (pure numpy, no GPU)
        def sample_pixel_data(mask, score, ratio=0.01):
            n = len(mask)
            max_pixel = int(n * ratio)
            if max_pixel < 100000:
                max_pixel = 100000
            if n <= max_pixel:
                return mask, score

            # Stratify and split positive/negative pixels
            pos_idx = np.nonzero(mask == 1)[0]
            neg_idx = np.nonzero(mask == 0)[0]
            pos_num = len(pos_idx)
            neg_num = len(neg_idx)
            # Allocate the number of samples according to the original ratio
            pos_sample = int(max_pixel * pos_num / n)
            neg_sample = max_pixel - pos_sample
            # Prevent the sample count from exceeding the actual number of samples
            pos_sample = min(pos_sample, pos_num)
            neg_sample = min(neg_sample, neg_num)

            # Random sampling without replacement
            sel_pos = np.random.choice(pos_idx, size=pos_sample, replace=False)
            sel_neg = np.random.choice(neg_idx, size=neg_sample, replace=False)
            sel = np.concatenate([sel_pos, sel_neg])
            return mask[sel], score[sel]
        
        p_mask  = np.concatenate(all_pix_masks)
        p_pred  = np.concatenate(all_pix_scores)  
        
        # New: stratified sampling, capped at 200k pixels max, pure CPU numpy
        p_mask_samp, p_pred_samp = sample_pixel_data(p_mask, p_pred, ratio=config.get("ratio", 0.01))

        # Compute pixel metrics using the sampled subset
        p_auroc = 100 * roc_auc_score(p_mask_samp, p_pred_samp)
        p_aupr  = 100 * average_precision_score(p_mask_samp, p_pred_samp)        
        pp, pr, _ = precision_recall_curve(p_mask_samp, p_pred_samp)
        p_f1max = 100 * np.max(2 * pp * pr / (pp + pr + 1e-8))
        
        pix_metric_cost = time.time() - pix_metric_start
  
        log_map = OrderedDict([
            ("dataset"             , dataset),
            ("model_filename"      , model_filename),
            
            ("sample_method"       , sample_method),
            ("sample_layer"        , sample_layer),
            ("k_shot"              , k_shot),
            
            ("few_shot_weight"     , few_shot_weight),
            ("logits_fusion"       , config.get("logits_fusion", "mean__layer_fusion")),            
            
            ("category"            , cat),
            ("img_count"           , img_count),
            ("selected_img_list"   , sel_img_list),
            
            ("meta_total_time_s"   , timings.get("meta_total_time", 0)),
            ("meta_img_count"      , timings.get("meta_img_count", 0)),
            ("meta_per_img_s"      , timings.get("meta_per_img_s", 0)),
            ("adapt_dec_time_s"    , timings.get("adapt_dec_time", 0)),
            ("coarse_time_s"       , timings.get("coarse_time", 0)),
            ("feat_extract_time_s" , timings.get("feat_extract_time", 0)),
            ("filter_time_s"       , timings.get("filter_time", 0)),
            
            ("mem_build_time_s"    , timings.get("mem_build_time", 0)),
            ("mem_load_time_s"     , timings.get("mem_load_time", 0)),
            
            ("I_AUROC"             , f"{i_auroc:.2f}"),
            ("I_AUPR"              , f"{i_aupr:.2f}"),
            ("I_F1max"             , f"{i_f1max:.2f}"),
            ("P_AUROC"             , f"{p_auroc:.2f}"),
            ("P_AUPR"              , f"{p_aupr:.2f}"),
            ("P_F1max"             , f"{p_f1max:.2f}"),
            ("category_time_s"     , f"{cat_time:.2f}"),
            ("avg_time_per_img_s"  , f"{avg_time:.3f}"),
            ("img_metric_time_s"   , f"{img_metric_cost:.3f}"),
            ("pix_metric_time_s"   , f"{pix_metric_cost:.1f}"),
            ("model_mem_GB"        , f"{base_mem:.2f}"),
            ("infer_mem_GB"        , f"{infer_mem:.2f}"),
        ])        
        
        if verbose:
            print("\n" + "=" * 55)
            for k, v in log_map.items():
                print(f"{k:<20}: {v}")
            print("=" * 55 + "\n")
        
        _, _, csv_path, _ = _get_dataset_config(config)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exist = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exist:
                writer.writerow(log_map.keys())
            writer.writerow(log_map.values())
            
    del memory_bank, test_loader, test_dataset, all_img_scores, all_img_labels, all_pix_scores, all_pix_masks
    torch.cuda.empty_cache()
    gc.collect()

# ===================== Main flow wrapper =====================
def main(config=config):
    checkpoint_dir = config["checkpoint_dir"]
    
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    model_files.sort()
    model_filenames = [os.path.splitext(f)[0] for f in model_files]
    
    print(f"\n")
    print(f"Total models: {len(model_filenames)}")
    print(f"Evaluation data: {config['dataset']}")
    print(f"Category list: {config['categories']}")

    for model_filename in model_filenames:
        config["model_filename"] = model_filename
        print(f"\n===== Starting evaluation of model: {model_filename} =====")
        
        checkpoint_path = f"{checkpoint_dir}/{model_filename}.pth"
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        state_dict = ckpt["detector"]
        t_cfg      = ckpt["config"]        
        
        # Config that must be consistent with training
        config["backbone"]      = t_cfg["backbone"]
        config["img_size"]      = t_cfg["img_size"]
        config["embed_dim"]     = t_cfg["embed_dim"]
        config["feat_layers"]   = t_cfg["feat_layers"]
        config["logits_fusion"] = t_cfg["logits_fusion"]
        
        # ================= Unified backbone loading =================
        backbone = BackboneWrapper(config=config).to(device).eval()
        
        detector = Detector(config=config).to(device).eval()
        
        detector.load_state_dict(state_dict, strict=False)
        # Loop inference over multiple layers, shots, weights, and categories
        for k_shot in config["k_shot"]:
            for cat in config["categories"]:
                for few_shot_weight in config["few_shot_weight"]:
                    evaluate(backbone, detector, k_shot, few_shot_weight, cat, config)

# ===================== Command-line entry: JSON loading + parameter override =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="LUMIN unified inference script, JSON base config + command-line parameter override, supporting CLIP/DINO dual-backbone batch evaluation"
    )
    # Hardware and random seed
    parser.add_argument("--cuda", type=str, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, help="global random seed")

    # Backbone configuration
    parser.add_argument("--backbone_name", type=str, choices=["CLIP", "DINO"], help="backbone type")
    parser.add_argument("--clip_name", type=str, help="CLIP model name")
    parser.add_argument("--dino_path", type=str, help="DINO weights directory")
    parser.add_argument("--feat_layers", nargs="+", type=int, help="list of feature layers to extract")

    # Dataset
    parser.add_argument("--dataset", type=str, choices=["mvtec","visa","btad","ksdd","real-iad"], help="target dataset")
    parser.add_argument("--split", type=str)
    parser.add_argument("--categories", nargs="+", type=str, help="specify the list of categories to evaluate")

    # Memory bank sampling parameters
    parser.add_argument("--sample_method", type=str, help="sampling strategy name")
    parser.add_argument("--sample_layer", nargs="+", type=int, help="list of single-layer indices for the memory bank")
    parser.add_argument("--k_shot", nargs="+", type=int, help="list of k-shot sample counts")
    parser.add_argument("--few_shot_weight", nargs="+", type=float, help="list of few-shot fusion weights")

    # Runtime hyperparameters
    parser.add_argument("--batch_size", type=int, help="inference batch size")
    parser.add_argument("--num_workers", type=int, help="number of dataloader worker processes")

    # Path outputs
    parser.add_argument("--checkpoint_dir", type=str, help="detector weights directory")
    parser.add_argument("--memory_bank_dir", type=str, help="memory bank cache root directory")
    parser.add_argument("--csv_path", type=str, help="root directory for saving metrics CSV")
    parser.add_argument("--vis_results_dir", type=str, help="root directory for saving visualization images")
    parser.add_argument("--metadata_cache_dir", type=str, help="sampling metadata cache directory")

    # Visualization and metrics toggles
    parser.add_argument("--vis", type=int, choices=[0,1], help="whether to visualize 0/1")
    parser.add_argument("--vis_threshold", type=float, help="visualization threshold")
    parser.add_argument("--metrics", type=int, choices=[0,1], help="whether to compute and save metrics 0/1")

    args = parser.parse_args()

    overwrite_map = {
        "cuda": "cuda",
        "seed": "seed_value",
        
        "backbone_name": "backbone",
        "clip_name": "clip_backbone",
        "dino_path": "dino_backbone",
        "feat_layers": "feat_layers",
        
        "dataset": "dataset",
        "split": "split",
        "categories": "categories",
        
        "sample_method": "sample_method",
        "sample_layer": "sample_layer",
        "k_shot": "k_shot",
        "few_shot_weight": "few_shot_weight",
        
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        
        "checkpoint_dir": "checkpoint_dir",
        "memory_bank_dir": "memory_bank_dir",
        "csv_path": "csv_path",
        "vis_results_dir": "vis_results_dir",
        "metadata_cache_dir": "metadata_cache_dir",
        
        "vis": "vis",
        "vis_threshold": "vis_threshold",
        "metrics": "metrics",
    }

    for arg_key, cfg_key in overwrite_map.items():
        arg_val = getattr(args, arg_key)
        if arg_val is not None:

            # Override the config dictionary
            config[cfg_key] = arg_val

    cuda_id = config["cuda"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    device = torch.device("cuda")

    # Start training
    main(config=config)