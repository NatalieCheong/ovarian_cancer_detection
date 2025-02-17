import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import resnet101, ResNet101_Weights

class OutlierHistoPathModel(nn.Module):
    def __init__(self, num_classes=5, feature_dim=512):
        super().__init__()
        # Base backbones (same as before)
        self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

        # Remove original classifier layers
        self.resnet.fc = nn.Identity()
        self.efficientnet.classifier = nn.Identity()

        # Feature dimensions
        self.resnet_dim = 2048
        self.efficient_dim = 1536
        self.feature_dim = feature_dim
        self.combined_resnet_dim = self.resnet_dim + (512 * 3)

        # Feature reduction layers (same as before)
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

        # Modify first conv layer
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Attention modules (same as before)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

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


        # Feature processors (same as before)
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


        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, num_classes)
        )

        # Outlier detection head
        self.outlier_detector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, self.feature_dim * 2)  # Mean and log variance
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
        x = self.resnet.layer4(x)

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

    def compute_outlier_score(self, x):
        """Compute outlier score for input images"""
        with torch.no_grad():
            # Extract features
            features = self.extract_features(x)

            # Get distribution parameters
            dist_params = self.outlier_detector(features)
            mean, log_var = torch.chunk(dist_params, 2, dim=1)

            # Compute Mahalanobis distance as outlier score
            var = torch.exp(log_var)
            z_score = (features - mean) / torch.sqrt(var + 1e-6)
            outlier_score = torch.sum(z_score ** 2, dim=1)

            return outlier_score

    def forward(self, x):
        """Forward pass with both classification and outlier detection"""
        # Extract features
        features = self.extract_features(x)

        # Classification logits
        logits = self.classifier(features)

        # Outlier detection parameters
        dist_params = self.outlier_detector(features)
        mean, log_var = torch.chunk(dist_params, 2, dim=1)

        if self.training:
            return logits, mean, log_var
        else:
            return logits

    def __del__(self):
        torch.cuda.empty_cache()

class OutlierLoss(nn.Module):
    """Combined loss for classification and outlier detection"""
    def __init__(self, num_classes=5, outlier_weight=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.outlier_weight = outlier_weight
        
    def forward(self, logits, mean, log_var, labels):
        # Classification loss
        ce_loss = self.ce_loss(logits, labels)
        
        # Feature distribution regularization
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        
        # Combined loss
        total_loss = ce_loss + self.outlier_weight * kl_loss
        
        return total_loss, ce_loss, kl_loss


def create_outlier_model(device, learning_rate=5e-4, epochs=30):
    """Create model with outlier detection capabilities"""
    model = OutlierHistoPathModel(num_classes=5)
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
    
    # Combined loss function
    criterion = OutlierLoss(num_classes=5)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=28,
        epochs=epochs,
        pct_start=0.3
    )
    
    return model, criterion, optimizer, scheduler, predictor, wsi_config
