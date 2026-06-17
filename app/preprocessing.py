"""
Image preprocessing pipeline for Diabetic Retinopathy detection.
Implements the exact preprocessing used during model training.
"""

import cv2
import io
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Tuple


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Target input size for the hybrid model
TARGET_SIZE = 224


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Crop out black borders from an image.
    
    This function crops the maximum rectangular area that doesn't 
    contain gray or nearly black pixels (useful for fundus images).
    
    Args:
        img: Input image as numpy array (grayscale or RGB)
        tol: Tolerance level for considering pixels as "black"
        
    Returns:
        Cropped image
    """
    if len(img.shape) == 2:
        mask = img > tol
        if not mask.any():
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
        if not mask.any():
            return img
        check_axis = np.ix_(mask.any(1), mask.any(0))
        return img[check_axis]
    else:
        return img


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, 
                tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    This enhances the local contrast of the image, which is particularly
    useful for retinal fundus images.
    
    Args:
        img: Input image as numpy array (BGR format)
        clip_limit: Clipping limit for the histogram
        tile_size: Size of the grid for histogram calculation
        
    Returns:
        CLAHE-enhanced image
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def preprocess_to_rgb_array(image_input) -> np.ndarray:
    """
    Prepare a retinal image as a 224x224 RGB uint8 array.
    
    Pipeline:
    1. Load image and convert to RGB
    2. Crop black borders
    3. Apply CLAHE enhancement
    4. Resize to target size
    
    Args:
        image_input: Either a file path (str), PIL Image, or numpy array
        
    Returns:
        Preprocessed image as RGB numpy array with shape (224, 224, 3)
        
    Raises:
        ValueError: If image cannot be loaded or processed
    """
    try:
        # Step 1: Load image and convert to RGB
        if isinstance(image_input, str):
            # Load from file path
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError("Cannot read image from file path")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to numpy array
            img = np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            # Assume numpy array is in BGR format
            if len(image_input.shape) == 2:
                # Grayscale - convert to RGB
                img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            else:
                # Assume BGR - convert to RGB
                img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Unsupported image input type")
        
        # Step 2: Crop black borders
        img = crop_image_from_gray(img)
        
        # Step 3: Apply CLAHE enhancement (convert to BGR for OpenCV)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = apply_clahe(img_bgr)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Step 4: Resize to target size
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), 
                         interpolation=cv2.INTER_LANCZOS4)

        return img.astype('uint8')

    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {str(e)}")


def tensor_from_rgb_array(img: np.ndarray) -> torch.Tensor:
    """
    Convert a preprocessed RGB image array into a normalized model tensor.
    """
    try:
        if img.shape[:2] != (TARGET_SIZE, TARGET_SIZE):
            img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE),
                             interpolation=cv2.INTER_LANCZOS4)
        
        # Convert from numpy (H, W, C) to PIL Image then to tensor
        img_pil = Image.fromarray(img.astype('uint8'), 'RGB')
        
        # Define normalization transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # Apply transform
        tensor = transform(img_pil)
        
        # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
        tensor = tensor.unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        raise ValueError(f"Error during tensor conversion: {str(e)}")


def preprocess_image(image_input) -> torch.Tensor:
    """
    Complete preprocessing pipeline for retinal fundus images.

    Returns:
        Preprocessed image as PyTorch tensor with shape (1, 3, 224, 224)
    """
    img = preprocess_to_rgb_array(image_input)
    return tensor_from_rgb_array(img)


def preprocess_from_bytes(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocess image directly from bytes.
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        Preprocessed image as PyTorch tensor
        
    Raises:
        ValueError: If image cannot be processed
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return preprocess_image(img)
    except Exception as e:
        raise ValueError(f"Error processing image from bytes: {str(e)}")


def preprocess_image_and_array_from_bytes(image_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image bytes for inference and return the display image used for Grad-CAM.

    Returns:
        Tuple of (model tensor, preprocessed RGB image array)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        display_img = preprocess_to_rgb_array(img)
        return tensor_from_rgb_array(display_img), display_img
    except Exception as e:
        raise ValueError(f"Error processing image from bytes: {str(e)}")
