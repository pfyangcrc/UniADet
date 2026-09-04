"""
==============================================================================
Unified backbone wrapper — BackboneWrapper (supports both CLIP / DINO models)
==============================================================================
The model type is switched via the config["backbone"] field, exposing a unified
extract_features interface so that downstream inference/training code does not
need to be aware of the underlying backbone differences.

config field specification:
    backbone:       "CLIP" | "DINO"
    clip_backbone:  CLIP model name (e.g. "ViT-L/14@336px")
    dino_backbone:  DINO local weights directory path
    embed_dim:      feature dimension (fixed 1024)
    feat_layers:    list of multi-scale feature layer indices (e.g. [11,14,17,20,23])

extract_features(x, sampling=False, sample_layer=11) → List[torch.Tensor]
    sampling=False: returns the [CLS + patch] concatenated features of the feat_layers layers
    sampling=True:  returns only the patch tokens of the specified sample_layer (no CLS)
==============================================================================
"""

import torch
import torch.nn as nn
from typing import List


class BackboneWrapper(nn.Module):
    # ---- CLIP default configuration ----
    CLIP_IMG_SIZE = 518
    
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

    # ---- DINO default configuration ----
    DINO_IMG_SIZE = 512
    
    DINO_MEAN = [0.485, 0.456, 0.406]
    DINO_STD  = [0.229, 0.224, 0.225]

    def __init__(self, config: dict):
        """
        Args:
            config: dictionary, must contain:
                - backbone: "CLIP" | "DINO"
                - embed_dim: 1024
                - feat_layers: List[int]
                - depending on the backbone type:
                  CLIP:  clip_backbone (str), clip_img_size (int)
                  DINO:  dino_backbone (str), dino_img_size (int)

        Raises:
            ValueError: invalid backbone value
            KeyError: missing required config field
        """
        super().__init__()

        # ---- Read common configuration ----
        backbone_type = config.get("backbone")
        if backbone_type is None:
            raise KeyError("config is missing the required field 'backbone' (allowed values: 'CLIP' | 'DINO')")

        self.model_type = backbone_type
        self.embed_dim = config.get("embed_dim", 1024)
        self.feat_layers = config.get("feat_layers", [11, 14, 17, 20, 23])
        self.feature_layers = self.feat_layers  # kept for backward compatibility with old code references

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ==================================================================
        # CLIP branch
        # ==================================================================
        if backbone_type == "CLIP":
            clip_backbone = config.get("clip_backbone")
            if clip_backbone is None:
                raise KeyError("backbone='CLIP' requires config['clip_backbone'] (e.g. 'ViT-L/14@336px')")

            self.image_size = config.get("clip_img_size", 518)

            import clip
            model, _ = clip.load(clip_backbone, device=device)
            self.model = model.float()
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        # ==================================================================
        # DINO branch
        # ==================================================================
        elif backbone_type == "DINO":
            dino_backbone = config.get("dino_backbone")
            if dino_backbone is None:
                raise KeyError("backbone='DINO' requires config['dino_backbone'] (path to a local weights directory)")

            self.image_size = config.get("dino_img_size", 512)

            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                dino_backbone,
                local_files_only=True,
                trust_remote_code=True
            ).to(device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        else:
            raise ValueError(
                f"Unsupported backbone type: '{backbone_type}', allowed values: 'CLIP' | 'DINO'"
            )

    # ======================================================================
    # CLIP: feature extraction (manually iterate through the transformer layers, supports positional embedding interpolation)
    # ======================================================================
    def _extract_clip(self, x, sampling: bool, sample_layer: int) -> List[torch.Tensor]:
        features = []
        with torch.no_grad():
            # conv1 projection
            x = self.model.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

            # Concatenate the CLS token
            cls_token = self.model.visual.class_embedding.to(x.dtype)
            cls_token = cls_token.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = torch.cat([cls_token, x], dim=1)

            # Positional embedding interpolation (adapts to arbitrary input sizes)
            pos_embed = self.model.visual.positional_embedding.to(x.dtype)
            if x.shape[1] != pos_embed.shape[0]:
                cls_embed = pos_embed[:1, :]
                patch_embed = pos_embed[1:, :]
                N = patch_embed.shape[0]
                orig_size = int(N ** 0.5)
                new_size = int((x.shape[1] - 1) ** 0.5)
                patch_embed = patch_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                patch_embed = torch.nn.functional.interpolate(
                    patch_embed, size=(new_size, new_size),
                    mode='bicubic', align_corners=False
                )
                patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(-1, patch_embed.shape[1])
                pos_embed = torch.cat([cls_embed, patch_embed], dim=0)

            x = x + pos_embed
            x = self.model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # [seq_len, B, dim]

            if sampling:
                for i, resblock in enumerate(self.model.visual.transformer.resblocks):
                    x = resblock(x)
                    if i == sample_layer:
                        feat = x.permute(1, 0, 2)           # [B, seq_len, dim]
                        img_tokens = feat[:, 1:, :]         # drop the CLS, keep only patches
                        features.append(img_tokens)
                        break
            else:
                for i, resblock in enumerate(self.model.visual.transformer.resblocks):
                    x = resblock(x)
                    if i in self.feature_layers:
                        feat = x.permute(1, 0, 2)           # [B, seq_len, dim]
                        cls_token_feat = feat[:, 0:1, :]    # CLS
                        img_tokens = feat[:, 1:, :]         # patch
                        feat = torch.cat([cls_token_feat, img_tokens], dim=1)
                        features.append(feat)

        return features

    # ======================================================================
    # DINO: feature extraction (transformers AutoModel forward pass, hidden_states indexing)
    # ======================================================================
    def _extract_dino(self, x, sampling: bool, sample_layer: int) -> List[torch.Tensor]:
        features = []
        with torch.no_grad():
            outputs = self.model(x, output_hidden_states=True)

            if sampling:
                feat = outputs.hidden_states[sample_layer]   # [B, 1029, 1024]
                img_tokens = feat[:, 5:, :]                  # first 4 special tokens + CLS, patches start at index 5
                features.append(img_tokens)
            else:
                for i in self.feature_layers:
                    feat = outputs.hidden_states[i]          # [B, 1029, 1024]
                    cls_token = feat[:, 0:1, :]              # CLS
                    img_tokens = feat[:, 5:, :]              # patch tokens
                    feat = torch.cat([cls_token, img_tokens], dim=1)  # [B, 1025, 1024]
                    features.append(feat)

        return features

    # ======================================================================
    # Unified public interface
    # ======================================================================
    def extract_features(self, x, sampling: bool = False, sample_layer: int = 11) -> List[torch.Tensor]:
        """Unified feature extraction entry point.

        Args:
            x: torch.Tensor [B, 3, H, W] input image batch
            sampling: when True, returns only the patch tokens of a single layer (used for memory bank sampling)
                      when False, returns the [CLS+patch] concatenated features of each feat_layers layer
            sample_layer: layer index to extract from when sampling=True

        Returns:
            List[torch.Tensor], each element has shape:
                sampling=False: [B, 1+num_patches, embed_dim]
                sampling=True:  [B, num_patches, embed_dim]
        """
        if self.model_type == "CLIP":
            return self._extract_clip(x, sampling, sample_layer)
        else:  # DINO
            return self._extract_dino(x, sampling, sample_layer)
