# -*- coding: utf-8 -*-
"""
m6_models.py

ONCOLENS - MODULE 6: NEURAL NETWORK MODELS
============================================
Model definitions for cell classification.

Contains:
    - ResNet+LSTM (96.58% accuracy) - RECOMMENDED
    - 3D CNN ResNet3D-18 (94.01% accuracy)

Save as: D:/Research/Cancer_Cell_Analysis/OncoLens/modules/m6_models.py

Author: Antardip Himel
Date: March 2026
"""

import torch
import torch.nn as nn
import torchvision.models as models

from . import m1_config as config


# =============================================================================
# RESNET + LSTM MODEL (Best: 96.58%)
# =============================================================================

class ResNetLSTM(nn.Module):
    """
    ResNet-18 + Bidirectional LSTM for video classification.
    
    Architecture:
        - ResNet-18 (modified for grayscale input) extracts spatial features
        - BiLSTM processes temporal sequence of 30 frames
        - Classifier outputs class predictions
    
    Input: (batch, num_frames, 1, 96, 96)
    Output: (batch, num_classes)
    
    Performance: 96.58% test accuracy, AUC 0.988
    """
    
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2, 
                 dropout=0.5, pretrained=False):
        super(ResNetLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Load ResNet-18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
            # Modify first conv for grayscale (average pretrained weights)
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                resnet.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        else:
            resnet = models.resnet18(weights=None)
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Get feature dimension and remove FC layer
        self.feature_dim = resnet.fc.in_features  # 512
        resnet.fc = nn.Identity()
        self.resnet = resnet
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classifier
        lstm_output_size = hidden_size * 2  # Bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape for ResNet: (batch * frames, C, H, W)
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features
        features = self.resnet(x)  # (batch * frames, 512)
        
        # Reshape for LSTM: (batch, frames, 512)
        features = features.view(batch_size, num_frames, -1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Combine forward and backward hidden states from last layer
        h_forward = h_n[-2, :, :]  # Last layer forward
        h_backward = h_n[-1, :, :]  # Last layer backward
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        
        # Classify
        logits = self.classifier(h_combined)
        
        return logits


# =============================================================================
# 3D CNN MODEL (ResNet3D-18) - 94.01%
# =============================================================================

class ResBlock3D(nn.Module):
    """3D Residual Block."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return self.relu(out)


class ResNet3D18(nn.Module):
    """
    3D ResNet-18 for video classification.
    
    Architecture:
        - 3D convolutions process spatial and temporal dimensions together
        - Residual blocks for deeper features
        - Global average pooling + classifier
    
    Input: (batch, 1, 30, 96, 96) - (batch, C, D, H, W)
    Output: (batch, num_classes)
    
    Performance: 94.01% test accuracy, AUC 0.983
    """
    
    def __init__(self, num_classes=2, dropout=0.5):
        super(ResNet3D18, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 7, 7),
                               stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))
        
        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        layers = []
        layers.append(ResBlock3D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return self.classifier(x)


# =============================================================================
# MODEL LOADING UTILITIES
# =============================================================================

def load_model(model_type, model_path=None, device=None):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_type: 'resnet_lstm' or '3dcnn'
        model_path: Path to checkpoint file (default: from config)
        device: Torch device (default: from config)
        
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = config.DEVICE
    
    if model_path is None:
        model_path = config.get_model_path(model_type)
    
    # Create model
    if model_type == 'resnet_lstm':
        model = ResNetLSTM(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.RESNET_LSTM_CONFIG['hidden_size'],
            num_layers=config.RESNET_LSTM_CONFIG['num_layers'],
            dropout=config.RESNET_LSTM_CONFIG['dropout']
        )
    elif model_type == '3dcnn':
        model = ResNet3D18(
            num_classes=config.NUM_CLASSES,
            dropout=config.CNN3D_CONFIG['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set eval mode
    model = model.to(device)
    model.eval()
    
    return model


def get_model_info(model_path):
    """
    Get information about a saved model checkpoint.
    
    Args:
        model_path: Path to checkpoint file
        
    Returns:
        dict with model info (epoch, val_acc, exists)
    """
    import os
    
    if not os.path.exists(model_path):
        return {
            'exists': False,
            'error': 'Model file not found'
        }
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        return {
            'exists': True,
            'epoch': checkpoint.get('epoch', -1) + 1,
            'val_acc': checkpoint.get('val_acc', 0),
            'val_acc_pct': f"{checkpoint.get('val_acc', 0) * 100:.2f}%"
        }
    except Exception as e:
        return {
            'exists': True,
            'error': str(e)
        }


def count_parameters(model):
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable