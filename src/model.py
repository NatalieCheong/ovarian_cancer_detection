from tiatoolbox.models.architecture import vanilla
from torchvision import models, transforms
from torch import nn
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from typing import Dict, Optional
import timm
import PIL.Image
from typing import Optional, Union, Dict
from tiatoolbox.models.models_abc import ModelABC
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import resnet101, ResNet101_Weights
from tiatoolbox.models.engine.patch_predictor import PatchPredictor, IOPatchPredictorConfig
import gc
import math
import traceback
import warnings
warnings.filterwarnings("ignore")

class HistoPathModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Initialize both backbones
        self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

        # Remove original classifier layers
        self.resnet.fc = nn.Identity()
        self.efficientnet.classifier = nn.Identity()

        # Get feature dimensions
        self.resnet_dim = 2048  # ResNet101's output dimension
        self.efficient_dim = 1536  # EfficientNet-B3's output dimension
        self.feature_dim = 512

        # Calculate combined features dimension for ResNet
        # Global features (2048) + 3 processors (512 each) = 3584
        self.combined_resnet_dim = self.resnet_dim + (512 * 3)

        # Feature reduction layers
        self.resnet_reducer = nn.Sequential(
            nn.Linear(self.combined_resnet_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.efficient_reducer = nn.Sequential(
            nn.Linear(self.efficient_dim, self.feature_dim * 2),
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim * 2, self.feature_dim)
        )

        # Modify first conv layer for histology images
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.Sigmoid()
        )
        # Add channel attention for EfficientNet
        self.efficient_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1536, 512, kernel_size=1),  # 1536 for EfficientNet-B3
            nn.ReLU(),
            nn.Conv2d(512, 1536, kernel_size=1),
            nn.Sigmoid()
        )

        # Add fusion attention
        self.fusion_attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 16),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 16, self.feature_dim),
            nn.Sigmoid()
        )


        # Feature processors for ResNet features
        self.feature_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(2048, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(2048, 512, 5, padding=2),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
        ])

        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.LayerNorm(self.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced dropout for stability
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.LayerNorm(self.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim * 2, self.feature_dim)
        )


        # Add feature gates
        self.feature_gates = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 2),
            nn.Softmax(dim=1)
        )


        # Add a squeeze-excitation block
        self.se_block = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 16),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 16, self.feature_dim),
            nn.Sigmoid()
         )


        # Self-attention for global context
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Main classifier
        self.main_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, num_classes)
        )

        # Auxiliary classifier
        self.aux_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, num_classes)
        )

    def extract_resnet_features(self, x):
        # Initial layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # ResNet blocks
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)  # Shape: [B, 2048, H, W]

        # Apply attention
        spatial_weights = self.spatial_attention(x)
        channel_weights = self.channel_attention(x)
        attended_features = x * spatial_weights * channel_weights

        # Global features
        global_features = F.adaptive_avg_pool2d(attended_features, 1).flatten(1)

        # Process features through each processor
        processed_features = []
        for processor in self.feature_processors:
            features = processor(attended_features)
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
            processed_features.append(pooled)

        # Concatenate global and processed features
        combined_features = torch.cat([global_features] + processed_features, dim=1)

        # Reduce features
        return self.resnet_reducer(combined_features)

    def extract_efficient_features(self, x):
        features = self.efficientnet.features(x)

        # Apply channel attention
        channel_weights = self.efficient_channel_attention(features)
        features = features * channel_weights

        features = self.efficientnet.avgpool(features)
        features = torch.flatten(features, 1)
        return self.efficient_reducer(features)


    def extract_features(self, x):
        resnet_features = self.extract_resnet_features(x)
        torch.cuda.empty_cache()

        efficient_features = self.extract_efficient_features(x)
        torch.cuda.empty_cache()

        combined_features = torch.cat([resnet_features, efficient_features], dim=1)
        fused_features = self.feature_fusion(combined_features)

        # Residual connection
        if hasattr(self, 'feature_gates'):
            gates = self.feature_gates(combined_features)
            residual = resnet_features * gates[:, 0].unsqueeze(1) + efficient_features * gates[:, 1].unsqueeze(1)
            fused_features = fused_features + residual


        # Apply self-attention with gradient clipping
        attended_features, _ = self.self_attention(
            fused_features.unsqueeze(1),
            fused_features.unsqueeze(1),
            fused_features.unsqueeze(1)
        )

        return fused_features + 0.1 * attended_features.squeeze(1)


    def forward(self, x):
        # Extract combined features
        features = self.extract_features(x)

        if self.training:
            # During training, return both main and auxiliary outputs
            main_logits = self.main_classifier(features)
            aux_logits = self.aux_classifier(features)
            return main_logits, aux_logits
        else:
            # During inference, only return main classifier output
            return self.main_classifier(features)

    def __del__(self):
        # Clean up CUDA memory
        torch.cuda.empty_cache()

def create_model(device, learning_rate=5e-4):
    """Create model and associated components."""
    # Create enhanced model
    model = HistoPathModel(num_classes=5)
    model = model.to(device)

    # Create predictor configuration
    wsi_config = IOPatchPredictorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        patch_input_shape=[224, 224],
        stride_shape=[224, 224]
    )

    # Create patch predictor
    predictor = PatchPredictor(
        model=model,
        batch_size=32,
        num_loader_workers=4
    )

    # Loss function with class weights
    #class_weights = torch.tensor([1.0, 1.8, 2.2, 4.7, 4.8]).to(device)
    class_weights = torch.tensor([1.2, 2.0, 2.4, 4.8, 4.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=28,
        epochs=50,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )

    return model, criterion, optimizer, scheduler, predictor, wsi_config

def main():
    """Main function to test model architecture"""
    try:
        print("Testing Model Architecture...")

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Create model and get all components
        model, criterion, optimizer, scheduler, predictor, wsi_config = create_model(
            device, learning_rate=5e-4)

        # Print model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\nModel Parameters:")
        print(f"Total: {total_params:,}")
        print(f"Trainable: {trainable_params:,}")

        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224).to(device)

        print(f"\nInput shape: {x.shape}")

        # Set model to eval mode for testing
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            if isinstance(outputs, tuple):
                main_logits = outputs[0]  # Get main classifier output
                probs = F.softmax(main_logits, dim=1)
                print(f"Output shape: {main_logits.shape}")

                # Print mean probabilities for each class
                mean_probs = probs.mean(dim=0)
                print("\nMean class probabilities:")
                for i, p in enumerate(mean_probs):
                    print(f"Class {i}: {p:.4f}")
            else:
                print(f"Output shape: {outputs.shape}")

        print("\nChecking PatchPredictor configuration:")
        print(f"Batch size: {predictor.batch_size}")
        print(f"WSI config input shape: {wsi_config.patch_input_shape}")

        print("\nModel architecture test completed successfully!")

        # Clean up
        del model, predictor
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in model test: {str(e)}")
        traceback.print_exc()

        # Clean up even if there's an error
        try:
            del model, predictor
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

if __name__ == "__main__":
    main()
