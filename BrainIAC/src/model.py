import torch
import torch.nn as nn
from monai.networks.nets import ViT
import torch.nn.functional as F
import yaml

class ViTBackboneNet(nn.Module):
    def __init__(self, simclr_ckpt_path):
        super(ViTBackboneNet, self).__init__()
        
        #  ViT backbone
        self.backbone = ViT(
            in_channels=1,  #  single channel input
            img_size=(96,96,96),  
            patch_size=(16, 16, 16),
            hidden_size=768,  
            mlp_dim=3072,
            num_layers=12,
            num_heads=12, 
            save_attn=True,
        )
        
        # Load pretrained weights from SimCLR checkpoint
        ckpt = torch.load(simclr_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        
        # Extract only backbone weights 
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                # Remove "backbone." prefix
                new_key = key[9:]  # len("backbone.") = 9
                backbone_state_dict[new_key] = value
        
        # Load the backbone weights
        self.backbone.load_state_dict(backbone_state_dict, strict=True)
        print("Backbone weights loaded!!")

    def forward(self, x):
        # Get features from ViT backbone
        features = self.backbone(x)
        
        # features[0][:, 0] gets CLS token: [batch_size, hidden_dim]
        cls_token = features[0][:, 0]  # Shape: [batch_size, 768]
        
        return cls_token

class Classifier(nn.Module):
    def __init__(self, d_model=768, num_classes=1):  # d_model=768 for ViT-B, num_classes=1 for regression
        super(Classifier, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.fc(x)
        return x

class SingleScanModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(SingleScanModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x 
    

class SingleScanModelBP(nn.Module):
    def __init__(self, backbone, classifier):
        super(SingleScanModelBP, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        
        
        scan_features_list = []
        for scan_tensor_with_extra_dim in x.split(1, dim=1):
            # Squeeze out the channel_dim (dim=1) 
            squeezed_scan_tensor = scan_tensor_with_extra_dim.squeeze(1)
            feature = self.backbone(squeezed_scan_tensor)
            scan_features_list.append(feature)
        
        # scan_features_list now looks - [(B, 768), (B, 768)]
        
        # Stack these features (dim=1)
       
        stacked_features = torch.stack(scan_features_list, dim=1)
        
        #  mean pooling -> (batch_size, 768)
       
        merged_features = torch.mean(stacked_features, dim=1)
        
        merged_features = self.dropout(merged_features)
        output = self.classifier(merged_features)
        return output 
    
class SingleScanModelQuad(nn.Module):
    """
    Model for quad image classification that processes four images through 
    shared backbone and merges their features.
    """
    def __init__(self, backbone, classifier):
        super(SingleScanModelQuad, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 4, C, D, H, W) - quad images
        Returns:
            output: Classification output
        """
        # Extract individual images
        image1 = x[:, 0]  # (batch_size, C, D, H, W)
        image2 = x[:, 1] 
        image3 = x[:, 2] 
        image4 = x[:, 3] 
        
        # Process all images through shared backbone
        features1 = self.backbone(image1)  # (batch_size, embed_dim)
        features2 = self.backbone(image2)  
        features3 = self.backbone(image3)  
        features4 = self.backbone(image4)  
        
        # mean pooling
       
        stacked_features = torch.stack([features1, features2, features3, features4], dim=1)
        merged_features = torch.mean(stacked_features, dim=1)
        
        # dropout and classifier
        merged_features = self.dropout(merged_features)
        output = self.classifier(merged_features)
        return output 