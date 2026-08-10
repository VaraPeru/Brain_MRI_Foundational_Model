import os
import yaml
import torch
import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized, NormalizeIntensityd, EnsureTyped
)
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.savers import NiftiSaver


from segmentation_model import ViTUNETRSegmentationModel

def load_model_for_inference(config, state_dict):
    """
    Loads a ViTUNETRSegmentationModel for inference.
    """
    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path=config['pretrain']['simclr_checkpoint_path'],
        img_size=tuple(config['model']['img_size']),
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    )
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            k = k[len('model.'):]
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    return model.eval().cuda()

def preprocess_image(image_path, config):
    """
    Loads and preprocesses a single image for inference.
    """
    img_size = tuple(config['model']['img_size'])
    
    # val transforms 
    transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image']),
        Resized(keys=['image'], spatial_size=img_size, mode='trilinear'),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        EnsureTyped(keys=['image'])
    ])
    
    data = transforms({'image': image_path})
    return data['image'].unsqueeze(0).cuda(), data['image_meta_dict']

def generate_segmentation(model, image_tensor, config):
    """
    Runs inference and returns the segmentation mask.
    """
    with torch.no_grad():
        pred = sliding_window_inference(
            inputs=image_tensor,
            roi_size=tuple(config['model']['img_size']),
            sw_batch_size=config['training']['sw_batch_size'],
            predictor=model,
            overlap=0.5
        )
    
    # apply treshold
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    # decollate into a list of tensors
    pred = decollate_batch(pred)[0]
    return pred


def save_segmentation(segmentation_tensor, meta_dict, output_path):
    """
    Saves the segmentation mask to a file.
    """
   
    saver = NiftiSaver(output_dir=os.path.dirname(output_path), 
                       output_postfix="", 
                       output_ext=".nii.gz",
                       separate_folder=False)
    
    # add chanel dim if not present 
    if len(segmentation_tensor.shape) == 3:
        segmentation_tensor = segmentation_tensor.unsqueeze(0)

   
    saver.save(segmentation_tensor, meta_dict)
    
   
    original_filename = os.path.basename(meta_dict.get('filename_or_obj', 'segmentation'))
   
    original_filename_no_ext = original_filename.split('.')[0]
    
    saved_path = os.path.join(os.path.dirname(output_path), f"{original_filename_no_ext}.nii.gz")
    
   
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(saved_path, output_path)
    print(f"Segmentation saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate segmentation for a single image")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to segmentation checkpoint file")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image (.nii.gz)")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save segmentation output")
    parser.add_argument('--simclr_checkpoint_path', type=str, required=False, default=None, help="Override SimCLR checkpoint path from saved config")
    parser.add_argument('--gpu_device', type=str, required=False, default="0", help="GPU device to use")
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    
    # Load checkpoint and extract config and state dict
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    config = checkpoint['hyper_parameters']
    state_dict = checkpoint['state_dict']

    # Override backbone path 
    if args.simclr_checkpoint_path:
        print(f"Overriding SimCLR checkpoint path with: {args.simclr_checkpoint_path}")
        config['pretrain']['simclr_checkpoint_path'] = args.simclr_checkpoint_path

    
    os.makedirs(args.output_dir, exist_ok=True)

    print("1. Loading model...")
    model = load_model_for_inference(config, state_dict)

    print(f"2. Loading and preprocessing image: {args.image_path}...")
    image_tensor, meta_dict = preprocess_image(args.image_path, config)

    print("3. Generating segmentation...")
    segmentation_tensor = generate_segmentation(model, image_tensor, config)
    
   
    image_filename = os.path.basename(args.image_path)
    name, ext = os.path.splitext(image_filename)
    if ext == ".gz":
        name, ext2 = os.path.splitext(name)
        ext = ext2 + ext
    
    output_filename = f"{name}_seg{ext}"
    output_path = os.path.join(args.output_dir, output_filename)
    

    print(f"4. Saving segmentation to {output_path}...")
    save_segmentation(segmentation_tensor.cpu(), meta_dict, output_path)

    print("Done.") 