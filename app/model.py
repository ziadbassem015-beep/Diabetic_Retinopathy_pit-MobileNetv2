"""
Model loading and inference utilities for Diabetic Retinopathy classification.
"""

import torch
import torch.nn as nn
import timm
from torchvision import models
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# HYBRID MODEL ARCHITECTURE
# ============================================================================
# 
# PASTE YOUR ORIGINAL HYBRID MODEL ARCHITECTURE HERE
# 
# This should include your custom HybridModel class that combines:
# - MobileNetV2 (feature extraction)
# - PiT (Pooling-based Vision Transformer)
# - Classification head
#
# Example structure:
#
# class HybridModel(nn.Module):
#     def __init__(self, num_classes=5, ...):
#         super(HybridModel, self).__init__()
#         # Your MobileNetV2 backbone
#         # Your PiT transformer components
#         # Your classification head
#         
#     def forward(self, x):
#         # Your forward pass implementation
#         return logits
#
# ============================================================================

class HybridModel(nn.Module):
    """
    Hybrid MobileNetV2 + PiT-S classifier matching the saved checkpoint.

    The checkpoint stores keys under Model_Mobile, Model_pit, and fc, so the
    attribute names here intentionally match the training-time model.
    """

    def __init__(self, num_classes: int = 5):
        super(HybridModel, self).__init__()
        self.Model_Mobile = models.mobilenet_v2(weights=None)
        self.Model_Mobile.classifier = nn.Identity()

        self.Model_pit = timm.create_model(
            "pit_s_224",
            pretrained=False,
            num_classes=0,
        )

        self.fc = nn.Linear(1280 + self.Model_pit.num_features, num_classes)

    def forward(self, x):
        mobile_features = self.Model_Mobile(x)
        pit_features = self.Model_pit(x)
        features = torch.cat((mobile_features, pit_features), dim=1)
        return self.fc(features)


class ModelManager:
    """
    Manages model loading, inference, and device management.
    """
    
    def __init__(self, model_weights_path: str, device: str = None):
        """
        Initialize the model manager.
        
        Args:
            model_weights_path: Path to the model weights file (.pt)
            device: Device to load model on ('cpu' or 'cuda'). 
                   If None, automatically selects GPU if available.
                   
        Raises:
            FileNotFoundError: If model weights file doesn't exist
            RuntimeError: If model loading fails
        """
        self.model_weights_path = Path(model_weights_path)
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Check if weights file exists
        if not self.model_weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.model_weights_path}"
            )
        
        # Load model
        self.model = self._load_model()
        logger.info("Model loaded successfully")
    
    def _load_model(self) -> nn.Module:
        """
        Load model from weights file.
        
        Returns:
            Loaded model in eval mode
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Load weights
            checkpoint = torch.load(
                self.model_weights_path,
                map_location=self.device,
                weights_only=False  # Required for custom architectures
            )
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # If checkpoint is a dict with model_state_dict key
                model_state = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # Alternative checkpoint format
                model_state = checkpoint['state_dict']
            else:
                # Assume checkpoint is the state dict directly
                model_state = checkpoint
            
            # Create model instance and load weights
            model = HybridModel(num_classes=5)
            model.load_state_dict(model_state)
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {self.model_weights_path}: {str(e)}"
            )
    
    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on input tensor.
        
        Args:
            input_tensor: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output logits of shape (batch_size, 5)
        """
        # Ensure input is on the correct device
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        
        return output
    
    @torch.no_grad()
    def predict_proba(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions for input.
        
        Args:
            input_tensor: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Softmax probabilities of shape (batch_size, 5)
        """
        logits = self.predict(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities
    
    def get_device(self) -> str:
        """Get the device the model is running on."""
        return str(self.device)
