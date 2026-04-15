import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=4,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=5,
        **kwargs,
    ):
        super().__init__()
        _ = kwargs

        if img_size % patch_size != 0:
            raise ValueError(f'img_size ({img_size}) must be divisible by patch_size ({patch_size})')

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        grid_size = img_size // patch_size
        num_patches = grid_size * grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        batch_size = x.shape[0]

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        return self.head(self.forward_features(x))


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


def build_transformer(num_classes=5, **kwargs):
    patch_size = kwargs.pop('patch_size', 16)
    img_size = kwargs.pop('img_size', 224)
    in_chans = kwargs.pop('in_chans', 3)
    embed_dim = kwargs.pop('embed_dim', 192)
    depth = kwargs.pop('depth', 4)
    num_heads = kwargs.pop('num_heads', 3)
    mlp_ratio = kwargs.pop('mlp_ratio', 4)
    _ = kwargs.pop('qkv_bias', True)
    dropout = kwargs.pop('drop_path_rate', 0.1)
    _ = kwargs
    return Transformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        num_classes=num_classes,
    )
