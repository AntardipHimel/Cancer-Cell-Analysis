# -*- coding: utf-8 -*-
"""
4_5_model.py

RESNET + LSTM MODEL FOR CELL CLASSIFICATION
============================================

Architecture:
1. ResNet-18 (modified for grayscale) - Feature extraction per frame
2. LSTM (bidirectional) - Temporal pattern learning
3. FC layers - Classification into 3 classes

Input:  (batch, 30, 1, 96, 96) - 30 grayscale frames
Output: (batch, 3) - logits for 3 classes

Author: Antardip Himel
Date: March 2026
"""

import torch
import torch.nn as nn
import torchvision.models as models


# =============================================================================
# MODEL
# =============================================================================

class ResNetLSTM(nn.Module):
    """
    ResNet + LSTM for video/sequence classification.
    
    Architecture:
    ┌─────────────────────────────────────────────┐
    │  INPUT: (batch, 30, 1, 96, 96)              │
    └─────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────┐
    │  ResNet-18 Feature Extractor                │
    │  - Modified first conv: 1 channel input     │
    │  - Output: 512-dim per frame                │
    │  - Weights shared across all 30 frames      │
    └─────────────────────────────────────────────┘
                        ↓
              (batch, 30, 512)
                        ↓
    ┌─────────────────────────────────────────────┐
    │  Bidirectional LSTM                         │
    │  - Hidden size: 256                         │
    │  - 2 layers with dropout                    │
    │  - Output: 512-dim (256*2 bidirectional)    │
    └─────────────────────────────────────────────┘
                        ↓
              (batch, 512)
                        ↓
    ┌─────────────────────────────────────────────┐
    │  Classifier Head                            │
    │  - FC: 512 → 256                            │
    │  - ReLU + Dropout                           │
    │  - FC: 256 → 3 classes                      │
    └─────────────────────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────┐
    │  OUTPUT: (batch, 3) logits                  │
    └─────────────────────────────────────────────┘
    
    Args:
        num_classes: Number of output classes (default: 3)
        hidden_size: LSTM hidden size (default: 256)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.5)
        pretrained: Use pretrained ResNet weights (default: True)
    """
    
    def __init__(self, num_classes=3, hidden_size=256, num_layers=2, 
                 dropout=0.5, pretrained=True):
        super(ResNetLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # =====================================================================
        # RESNET FEATURE EXTRACTOR
        # =====================================================================
        
        # Load pretrained ResNet-18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)
        
        # Modify first conv layer for grayscale (1 channel instead of 3)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # New:      Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels=1,  # Grayscale
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Initialize new conv with mean of original RGB weights
        if pretrained:
            with torch.no_grad():
                resnet.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Remove the final FC layer (we'll add our own)
        self.feature_dim = resnet.fc.in_features  # 512 for ResNet-18
        resnet.fc = nn.Identity()
        
        self.resnet = resnet
        
        # =====================================================================
        # LSTM TEMPORAL MODULE
        # =====================================================================
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,  # 512
            hidden_size=hidden_size,       # 256
            num_layers=num_layers,         # 2
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # =====================================================================
        # CLASSIFIER HEAD
        # =====================================================================
        
        # Bidirectional LSTM outputs hidden_size * 2
        lstm_output_size = hidden_size * 2  # 512
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, num_frames, channels, height, width)
               e.g., (8, 30, 1, 96, 96)
        
        Returns:
            logits: Output tensor of shape (batch, num_classes)
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape to process all frames through ResNet at once
        # (batch, 30, 1, 96, 96) -> (batch*30, 1, 96, 96)
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features for all frames
        # (batch*30, 1, 96, 96) -> (batch*30, 512)
        features = self.resnet(x)
        
        # Reshape back to sequence
        # (batch*30, 512) -> (batch, 30, 512)
        features = features.view(batch_size, num_frames, -1)
        
        # Pass through LSTM
        # (batch, 30, 512) -> (batch, 30, 512) [bidirectional: 256*2]
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Use the last hidden state from both directions
        # h_n shape: (num_layers*2, batch, hidden_size) for bidirectional
        # Concatenate forward and backward final hidden states
        h_forward = h_n[-2, :, :]  # Last layer, forward
        h_backward = h_n[-1, :, :]  # Last layer, backward
        h_combined = torch.cat([h_forward, h_backward], dim=1)  # (batch, 512)
        
        # Classify
        logits = self.classifier(h_combined)
        
        return logits
    
    def get_features(self, x):
        """Extract features without classification (for visualization)."""
        batch_size, num_frames, channels, height, width = x.shape
        
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.resnet(x)
        features = features.view(batch_size, num_frames, -1)
        
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        
        return h_combined


# =============================================================================
# MODEL SUMMARY
# =============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model):
    """Print model summary."""
    print("=" * 70)
    print("  MODEL SUMMARY: ResNet-18 + LSTM")
    print("=" * 70)
    
    print(f"\n  Architecture:")
    print(f"    ResNet-18 feature dim: {model.feature_dim}")
    print(f"    LSTM hidden size: {model.hidden_size}")
    print(f"    LSTM layers: {model.num_layers}")
    print(f"    LSTM bidirectional: Yes")
    print(f"    Output classes: {model.num_classes}")
    
    print(f"\n  Parameters:")
    total_params = count_parameters(model)
    print(f"    Total trainable: {total_params:,}")
    
    # Count by component
    resnet_params = count_parameters(model.resnet)
    lstm_params = count_parameters(model.lstm)
    classifier_params = count_parameters(model.classifier)
    
    print(f"    ResNet-18: {resnet_params:,} ({resnet_params/total_params*100:.1f}%)")
    print(f"    LSTM: {lstm_params:,} ({lstm_params/total_params*100:.1f}%)")
    print(f"    Classifier: {classifier_params:,} ({classifier_params/total_params*100:.1f}%)")
    
    print("=" * 70)


# =============================================================================
# TEST MODEL
# =============================================================================

def test_model():
    """Test the model with dummy input."""
    
    print("=" * 70)
    print("  TESTING MODEL")
    print("=" * 70)
    
    # Create model
    print("\n  Creating model...")
    model = ResNetLSTM(num_classes=3, pretrained=True)
    
    # Print summary
    model_summary(model)
    
    # Test with dummy input
    print("\n  Testing forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    
    model = model.to(device)
    model.eval()
    
    # Dummy input: batch=2, frames=30, channels=1, height=96, width=96
    dummy_input = torch.randn(2, 30, 1, 96, 96).to(device)
    print(f"    Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"    Output shape: {output.shape}")
    print(f"    Output sample: {output[0].cpu().numpy()}")
    
    # Test predictions
    probs = torch.softmax(output, dim=1)
    preds = torch.argmax(probs, dim=1)
    print(f"    Predictions: {preds.cpu().numpy()}")
    
    print("\n" + "=" * 70)
    print("  ✅ MODEL TEST PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_model()

"""
(slowfast) PS D:\Research\Cancer_Cell_Analysis> & "C:\Users\Antardip Himel\.conda\envs\slowfast\python.exe" d:/Research/Cancer_Cell_Analysis/scripts/4_5_model.py
======================================================================
  TESTING MODEL
======================================================================

  Creating model...
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to C:\Users\Antardip Himel/.cache\torch\hub\checkpoints\resnet18-f37072fd.pth
100%|███████████████████████████████████████████████████████████████████████████████████████████████| 44.7M/44.7M [00:02<00:00, 19.6MB/s]
======================================================================
  MODEL SUMMARY: ResNet-18 + LSTM
======================================================================

  Architecture:
    ResNet-18 feature dim: 512
    LSTM hidden size: 256
    LSTM layers: 2
    LSTM bidirectional: Yes
    Output classes: 3

  Parameters:
    Total trainable: 14,456,259
    ResNet-18: 11,170,240 (77.3%)
    LSTM: 3,153,920 (21.8%)
    Classifier: 132,099 (0.9%)
======================================================================

  Testing forward pass...
    Device: cuda
    Input shape: torch.Size([2, 30, 1, 96, 96])
    Output shape: torch.Size([2, 3])
    Output sample: [ 0.03111659 -0.01031144  0.01455207]
    Predictions: [0 0]

======================================================================
  ✅ MODEL TEST PASSED!
======================================================================
(slowfast) PS D:\Research\Cancer_Cell_Analysis> 
"""