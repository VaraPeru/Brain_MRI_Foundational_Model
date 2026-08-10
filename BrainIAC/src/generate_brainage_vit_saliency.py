import os
import torch
import numpy as np
import nibabel as nib
import yaml
from monai.transforms import (
    Compose, LoadImaged, Resized, NormalizeIntensityd, ToTensord, ScaleIntensityd
)
from train_lightning_brainage import BrainAgeLightningModule

# ---- HARD CODED PATHS ----
nifti_path = ""
checkpoint_path = ""
config_path = ""
output_dir = ""
layer = -1  # Transformer layer index to visualize (-1 for last layer)
img_size = (96, 96, 96)  # Input image size (DxHxW), should match training
patch_size = 16  # ViT patch size, should match training
# -----------------------------------------

def get_preprocessing_transform(img_size):
    """Returns the MONAI preprocessing transforms for the input image."""
    return Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True),
        Resized(keys=["image"], spatial_size=img_size),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        #ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ToTensord(keys=["image"])
    ])

def extract_attention_map(vit_model, image, layer_idx=-1, img_size=(96, 96, 96), patch_size=16):
    """
    Extracts the attention map from a Vision Transformer (ViT) model.

    This function wraps the attention blocks of the ViT to capture the attention
    weights during a forward pass. It then processes these weights to generate
    a 3D saliency map corresponding to the model's focus on the input image.
    """
    attention_maps = {}

    #  wrapper class 
    class AttentionWithWeights(torch.nn.Module):
        def __init__(self, original_attn_module):
            super().__init__()
            self.original_attn_module = original_attn_module
            self.attn_weights = None

        def forward(self, x):
        
            output = self.original_attn_module(x)
            if hasattr(self.original_attn_module, 'qkv'):
                qkv = self.original_attn_module.qkv(x)
                batch_size, seq_len, _ = x.shape
                
                qkv = qkv.reshape(batch_size, seq_len, 3, self.original_attn_module.num_heads, -1)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * self.original_attn_module.scale
                self.attn_weights = attn.softmax(dim=-1)
            return output

    # Replace the attention module in each block with our wrapper
    for i, block in enumerate(vit_model.blocks):
        if hasattr(block, 'attn'):
            block.attn = AttentionWithWeights(block.attn)

    # Perform a forward pass 
    with torch.no_grad():
        _ = vit_model(image)

    # Collect the  attention weights from each block
    for i, block in enumerate(vit_model.blocks):
        if hasattr(block.attn, 'attn_weights') and block.attn.attn_weights is not None:
            attention_maps[f"layer_{i}"] = block.attn.attn_weights.detach()

    if not attention_maps:
        raise RuntimeError("Could not extract any attention maps. Please check the ViT model structure.")

    # Select the layer
    if layer_idx < 0:
        layer_idx = len(attention_maps) + layer_idx
    layer_name = f"layer_{layer_idx}"
    if layer_name not in attention_maps:
        raise ValueError(f"Layer {layer_idx} not found. Available layers: {list(attention_maps.keys())}")

    layer_attn = attention_maps[layer_name]
    # Average attention across all heads
    head_attn = layer_attn[0].mean(dim=0)
   
    cls_attn = head_attn[0, 1:]

    # Reshape  into a 3D volume
    patches_per_dim = img_size[0] // patch_size
    total_patches = patches_per_dim ** 3
    
   
    if cls_attn.shape[0] != total_patches:
        if cls_attn.shape[0] > total_patches:
            cls_attn = cls_attn[:total_patches]
        else:
            padded = torch.zeros(total_patches, device=cls_attn.device)
            padded[:cls_attn.shape[0]] = cls_attn
            cls_attn = padded

    cls_attn_3d = cls_attn.reshape(patches_per_dim, patches_per_dim, patches_per_dim)
    cls_attn_3d = cls_attn_3d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    # Upsample the attention map to the full image resolution
    upsampled_attn = torch.nn.functional.interpolate(
        cls_attn_3d,
        size=img_size,
        mode='trilinear',
        align_corners=False
    ).squeeze()

    # Normalize the map to [0, 1] for visualization
    upsampled_attn = upsampled_attn.cpu().numpy()
    upsampled_attn = (upsampled_attn - upsampled_attn.min()) / (upsampled_attn.max() - upsampled_attn.min())
    return upsampled_attn

def main():
    """Main function to load the model, generate, and save the saliency map."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load the checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = BrainAgeLightningModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location=device,
        strict=True 
    )
    model.eval()

    
    vit_model = model.backbone.backbone

    # Preprocess 
    transforms = get_preprocessing_transform(img_size)
    print(f"Loading and preprocessing image: {nifti_path}")
    image_dict = transforms({"image": nifti_path})
    image = image_dict["image"].unsqueeze(0).to(device)

    
    print("Extracting attention map...")
    attn_map = extract_attention_map(vit_model, image, layer_idx=layer, img_size=img_size, patch_size=patch_size)
    print("...extraction complete.")

    # Save the saliency map as NIfTI files
    base_filename = os.path.basename(nifti_path).split('.')[0]
    checkpoint_name = os.path.basename(checkpoint_path).split('.')[0]
    
    input_nifti = nib.Nifti1Image(image.cpu().squeeze().numpy(), np.eye(4))
    saliency_nifti = nib.Nifti1Image(attn_map, np.eye(4)) # Using identity affine
    
    input_save_path = os.path.join(output_dir, f"{base_filename}_{checkpoint_name}_input.nii.gz")
    saliency_save_path = os.path.join(output_dir, f"{base_filename}_{checkpoint_name}_saliencymap_layer{layer}.nii.gz")
    
    nib.save(input_nifti, input_save_path)
    nib.save(saliency_nifti, saliency_save_path)
    
    print(f"Successfully saved input image to: {input_save_path}")
    print(f"Successfully saved saliency map to: {saliency_save_path}")

if __name__ == "__main__":
    main() 