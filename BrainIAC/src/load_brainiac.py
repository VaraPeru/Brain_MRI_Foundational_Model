import torch
from model import ViTBackboneNet
import argparse

def load_brainiac(checkpoint_path, device='cuda'):
    """
    Load the ViT backbone model and BrainIAC checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        device (str): Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded ViT backbone model with checkpoint weights
    """
    # Create ViT backbone model - the constructor handles checkpoint loading
    model = ViTBackboneNet(checkpoint_path)
    
    # Move model to specified device
    model = model.to(device)
    
    return model

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description='Load ViT backbone model from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to load the model on (cuda or cpu)')
    args = parser.parse_args()
    
    # Load model
    model = load_brainiac(args.checkpoint, args.device)
    print(f"ViT backbone model loaded successfully from {args.checkpoint}!")