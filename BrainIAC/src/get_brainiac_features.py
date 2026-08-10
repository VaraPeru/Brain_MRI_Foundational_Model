import torch
import numpy as np
import pandas as pd
import random
import yaml
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import BrainAgeDataset, get_validation_transform
from load_brainiac import load_brainiac

# fix random seed 
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

#=========================
# Inference function
#=========================

def infer(model, test_loader):
    features_df = None  # Placeholder for feature DataFrame
    model.eval()
    
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Extracting ViT features", unit="batch"):
            inputs = sample['image'].to(device)
            class_labels = sample['label'].float().to(device)

            # Get features from the ViT backbone model
            features = model(inputs)
            features_numpy = features.cpu().numpy()

            # Expand features into separate columns
            feature_columns = [f'Feature_{i}' for i in range(features_numpy.shape[1])]
            batch_features = pd.DataFrame(
                features_numpy,
                columns=feature_columns
            )
            batch_features['GroundTruthClassLabel'] = class_labels.cpu().numpy().flatten()

            # Append batch features to features_df
            if features_df is None:
                features_df = batch_features
            else:
                features_df = pd.concat([features_df, batch_features], ignore_index=True)
    
    return features_df

#=========================
# Main inference pipeline
#=========================

def main():
    # argparse
    parser = argparse.ArgumentParser(description='Extract ViT BrainIAC features from images')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to the ViT BrainIAC model checkpoint (default: checkpoints/BrainIAC.ckpt)')
    parser.add_argument('--input_csv', type=str, required=True,
                      help='Path to the input CSV file containing image paths')
    parser.add_argument('--output_csv', type=str, required=True,
                      help='Path to save the output features CSV')
    parser.add_argument('--root_dir', type=str, required=True,
                      help='Root directory containing the image data')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for inference (default: 1)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of workers for data loading (default: 1)')
    args = parser.parse_args()

    # Setup dataset with validation transforms (no augmentation for feature extraction)
    test_dataset = BrainAgeDataset(
        csv_path=args.input_csv,
        root_dir=args.root_dir,
        transform=get_validation_transform()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Load ViT BrainIAC - device placement is handled in load_brainiac
    model = load_brainiac(args.checkpoint, device)
    
    # Extract features
    features_df = infer(model, test_loader)
    
    # Save features
    features_df.to_csv(args.output_csv, index=False)
    print(f"ViT BrainIAC features saved to {args.output_csv}")
    print(f"Feature shape: {features_df.shape}")
    print(f"Number of feature dimensions: {features_df.shape[1] - 1}")  # -1 for label column

if __name__ == "__main__":
    main()