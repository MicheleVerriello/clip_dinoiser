# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology & Oriane Simeoni, valeo.ai
# ---------------------------------------------------------------------------------------------------

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from mmseg.ops import resize
from omegaconf import OmegaConf

from models.builder import MODELS, build_model

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


@MODELS.register_module()
class DinoCLIP(nn.Module):
    """
    Base model for all the backbones. Implements CLIP features refinement based on DINO dense features and background
    refinement using FOUND model.

    """

    def __init__(self, clip_backbone, class_names, vit_arch="vit_base", vit_patch_size=16, enc_type_feats="k",
                 gamma=0.2, delta=0.99, apply_found=False):
        super(DinoCLIP, self).__init__()
        self.vit_arch = vit_arch
        self.enc_type_feats = enc_type_feats
        self.gamma = gamma
        self.vit_patch_size = vit_patch_size
        self.apply_found = apply_found
        self.delta = delta

        # ==== build MaskCLIP backbone =====
        maskclip_cfg = OmegaConf.load(f"configs/{clip_backbone}.yaml")
        self.clip_backbone = build_model(maskclip_cfg["model"], class_names=class_names)
        for param in self.clip_backbone.parameters():
            param.requires_grad = False

    def load_teachers(self):
        from models.FOUND.model import FoundModel, get_vit_encoder

        self.found_model = FoundModel()
        assert os.path.isfile("models/FOUND/data/weights/decoder_weights.pt"), 'No weights for FOUND model'
        self.found_model.decoder_load_weights("models/FOUND/data/weights/decoder_weights.pt")
        self.found_model.eval()
        for param in self.found_model.parameters():
            param.requires_grad = False

        # ==== build DINO backbone =====
        self.vit_encoder, self.initial_dim, self.hook_features = get_vit_encoder(
            self.vit_arch,
            "dino",
            self.vit_patch_size,
            self.enc_type_feats,
        )
        self.vit_encoder.eval()
        for param in self.vit_encoder.parameters():
            param.requires_grad = False

        # ==== build transform =====
        self.dino_T = NORMALIZE

    def make_input_divisible(self, x: torch.Tensor) -> torch.Tensor:
        """Pad some pixels to make the input size divisible by the patch size."""
        B, _, H_0, W_0 = x.shape
        pad_w = (self.vit_patch_size - W_0 % self.vit_patch_size) % self.vit_patch_size
        pad_h = (self.vit_patch_size - H_0 % self.vit_patch_size) % self.vit_patch_size
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x

    @torch.no_grad()
    def extract_feats(self, type_feats="k"):
        """
        DINO feature extractor. Attaches a hook on the last attention layer.
        :param type_feats: (string) - type of features from DINO ViT
        """
        nh = self.vit_encoder.blocks[-1].attn.num_heads
        nb_im, nb_tokens, C_qkv = self.hook_features["qkv"].shape

        qkv = (
            self.hook_features["qkv"]
                .reshape(
                nb_im, nb_tokens, 3, nh, C_qkv // nh // 3
            )  # 3 corresponding to |qkv|
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if type_feats == "q":
            return q.transpose(1, 2).float()
        elif type_feats == "k":
            return k.transpose(1, 2).float()
        elif type_feats == "v":
            return v.transpose(1, 2).float()
        else:
            raise ValueError("Unknown features")

    @torch.no_grad()
    def forward(self, x: torch.Tensor):

        x = self.make_input_divisible(x)
        output, raw_similarities = self.get_clip_features(x)
        B, _, H_feat, W_feat = output.shape
        masks = self.get_dino_corrs(x)
        output = self.compute_weighted_pool(output, masks)
        output = self.clip_backbone.decode_head.cls_seg(output)
        if self.apply_found:
            # Compute FOUND --------------------------------------------------
            preds = self.get_found_preds(x)
            r_soft_found = T.functional.resize(preds, (H_feat, W_feat)).reshape(-1)
            nb_cls = output.shape[1]
            r_hard_found = (r_soft_found > 0.5).float()
            uncertain = (output.max(dim=1)[0] < self.delta).reshape(-1)
            output.reshape(1, nb_cls, -1)[:, 0, uncertain & (~r_hard_found.bool())] = 1.0  # background class
        return output

    @torch.no_grad()
    def get_dino_corrs(self, x: torch.Tensor):
        """
        Gets correlations of DINO features. Applies a threshold on the correlations with self.gamma.

        :param x: (torch.Tensor) - batch of input images
        :return: (torch.Tensor) - feature correlations
        """
        B = x.shape[0]
        feats, (hf, wf) = self.get_dino_features(x)  # B C (h_f * w_f) normalized
        corrs = torch.matmul(feats.permute(0, 2, 1), feats).reshape(B, hf, wf, hf * wf)
        if self.gamma is not None:
            corrs[corrs < self.gamma] = 0.0

        return corrs.permute(0, 3, 1, 2)  # B C h w

    def get_dino_features(self, x: torch.Tensor):
        """
        Extracts dense DINO features.

        :param x: (torch.Tensor) - batch of input images
        :return: (torch.Tensor) - of dense DINO features, (int, int) - size of feature map
        """
        x = self.make_input_divisible(x)
        batch = self.dino_T(x)  # tensor B C H W
        h_featmap = batch.shape[-2] // self.vit_patch_size
        w_featmap = batch.shape[-1] // self.vit_patch_size

        # Forward pass
        # Encoder forward pass and get hooked intermediate values
        _ = self.vit_encoder(batch)

        # Get decoder features
        feats = self.extract_feats(type_feats=self.enc_type_feats)
        num_extra_tokens = 1

        # B nbtokens+1 nh dim
        feats = feats[:, num_extra_tokens:, :, :].flatten(-2, -1).permute(0, 2, 1)  # B C nbtokens
        # B, C, nbtokens
        feats = feats / feats.norm(dim=1, keepdim=True)  # normalize features

        return feats, (h_featmap, w_featmap)

    @torch.no_grad()
    def get_clip_features(self, x: torch.Tensor):
        """
        Extracts MaskCLIP features
        :param x: (torch.Tensor) - batch of input images
        :return: (torch.Tensor) - clip dense features, (torch.Tensor) - output probabilities
        """
        x = self.make_input_divisible(x)
        maskclip_map, feat = self.clip_backbone(x, return_feat=True)

        return feat, maskclip_map

    @torch.no_grad()
    def get_found_preds(self, x: torch.Tensor, resize=None):
        """
        Gets FOUND predictions.
        :param x: (torch.Tensor) - batch of input images
        :param resize: optional (tuple(int)) - size to resize the output prediction map to
        :return: (torch.Tensor) - saliency predictions
        """
        x = self.make_input_divisible(x)
        x = self.dino_T(x)
        found_preds, _, shape_f, att = self.found_model.forward_step(x, for_eval=True)
        preds = torch.sigmoid(found_preds.detach()).float()
        if resize is not None:
            preds = T.functional.resize(preds, resize)
        return preds

    @staticmethod
    def compute_weighted_pool(maskclip_feats: torch.Tensor, corrs: torch.Tensor):
        """
        Weighted pooling method.
        :param maskclip_feats: torch.tensor - raw clip features
        :param corrs: torch.tensor - correlations as weights for pooling mechanism
        :return: torch.tensor - refined clip features
        """
        B = maskclip_feats.shape[0]
        h_m, w_m = maskclip_feats.shape[-2:]
        h_w, w_w = corrs.shape[-2:]

        if (h_m != h_w) or (w_m != w_w):
            maskclip_feats = resize(
                input=maskclip_feats,
                size=(h_w, w_w),
                mode='bilinear',
                align_corners=False)
            h_m

    def set_support_images(self, support_images):
        """
        Imposta le immagini di supporto nel modello.
        """
        self.support_images = support_images  # Memorizza le immagini come attributo del modello