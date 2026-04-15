import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=5, base_channels=32, dropout=0.3, **kwargs):
        super().__init__()
        _ = kwargs

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class MetadataOnlyClassifier(nn.Module):
    def __init__(self, metadata_dim, num_classes):
        super().__init__()
        hidden = max(64, metadata_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, metadata_tensor):
        return self.net(metadata_tensor)


class IntermediateFusionClassifier(nn.Module):
    def __init__(self, image_model, metadata_dim, num_classes):
        super().__init__()
        self.image_model = image_model
        image_feature_dim = num_classes
        metadata_feature_dim = max(32, metadata_dim)

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, metadata_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(image_feature_dim + metadata_feature_dim, max(64, num_classes * 8)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(max(64, num_classes * 8), num_classes),
        )

    def forward(self, image_tensor, metadata_tensor):
        image_logits = self.image_model(image_tensor)
        metadata_features = self.metadata_encoder(metadata_tensor)
        fused = torch.cat([image_logits, metadata_features], dim=1)
        return self.fusion_head(fused)


class LateFusionClassifier(nn.Module):
    def __init__(self, image_model, metadata_dim, num_classes):
        super().__init__()
        self.image_model = image_model
        self.metadata_model = MetadataOnlyClassifier(metadata_dim, num_classes)

    def forward(self, image_tensor, metadata_tensor):
        image_logits = self.image_model(image_tensor)
        metadata_logits = self.metadata_model(metadata_tensor)
        return 0.5 * image_logits + 0.5 * metadata_logits


def build_cnn(num_classes=5, **kwargs):
    drop_path_rate = kwargs.pop('drop_path_rate', 0.15)
    dropout = min(0.5, max(0.0, float(drop_path_rate) + 0.15))
    base_channels = kwargs.pop('base_channels', 32)
    _ = kwargs
    return CNN(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
