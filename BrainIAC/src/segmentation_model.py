import torch
import torch.nn as nn
from monai.networks.nets import ViT, UNETR

class ViTUNETRSegmentationModel(nn.Module):
    def __init__(self, simclr_ckpt_path, img_size=(96,96,96), in_channels=1, out_channels=1):
        super().__init__()
        # Load ViT backbone
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=(16,16,16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=False,
        )
        # Load SimCLR weights
        ckpt = torch.load(simclr_ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        backbone_state_dict = {k[9:]: v for k, v in state_dict.items() if k.startswith('backbone.')}
        self.vit.load_state_dict(backbone_state_dict, strict=True)
        # UNETR decoder
        self.unetr = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name='instance',
            res_block=True,
            dropout_rate=0.0
        )
        
        # Transfer ViT weights to UNETR encoder
        self.unetr.vit.load_state_dict(self.vit.state_dict(), strict=True)
        print("="*10)
        print("ViT loaded from scratch")
        print("="*10)
    def forward(self, x):
        return self.unetr(x) 