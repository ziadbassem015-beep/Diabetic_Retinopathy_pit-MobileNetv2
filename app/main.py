"""
FastAPI application for Diabetic Retinopathy classification.

Serves predictions from a PyTorch hybrid model (MobileNetV2 + PiT).

Endpoints:
- GET /: Root endpoint with health status
- GET /health: Health check endpoint
- POST /predict: Image prediction endpoint
- POST /gradcam: Grad-CAM overlay endpoint
"""

import cv2
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response

from app.gradcam import generate_gradcam_overlay
from app.model import ModelManager
from app.preprocessing import preprocess_from_bytes, preprocess_image_and_array_from_bytes
from app.schemas import HealthResponse, PredictionResponse
from app.utils import get_label_from_class, format_probabilities


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global model manager - will be initialized on startup
model_manager = None
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/gif", "image/webp"}


def validate_model_ready():
    """Ensure the model is available before handling inference requests."""
    if model_manager is None:
        logger.error("Model manager not initialized")
        raise HTTPException(
            status_code=500,
            detail="Model not initialized. Please restart the server."
        )


def validate_upload(file: UploadFile):
    """Validate image uploads shared by prediction and Grad-CAM endpoints."""
    if file is None or file.size == 0:
        logger.warning("Empty file upload attempted")
        raise HTTPException(
            status_code=400,
            detail="No file uploaded or file is empty"
        )

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events.
    """
    # Startup
    global model_manager
    try:
        weights_dir = Path(__file__).parent.parent / "weights"
        model_weights_path = weights_dir / "best_model.pt"
        if not model_weights_path.exists():
            model_weights_path = weights_dir / "Mobile_pit_epoch_18.pt"

        logger.info(f"Loading model weights from {model_weights_path}")
        model_manager = ModelManager(str(model_weights_path))
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI application
app = FastAPI(
    title="Diabetic Retinopathy Classification API",
    description="REST API for classifying diabetic retinopathy severity from fundus images",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint - returns API status.
    
    Returns:
        Health status message
    """
    return {
        "message": "Diabetic Retinopathy Classification API",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health", tags=["Health"], response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with status
    """
    return HealthResponse(status="healthy")


@app.post("/predict", tags=["Prediction"], response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict Diabetic Retinopathy classification from fundus image.
    
    Accepts:
        - Image file upload (JPEG, PNG, etc.)
    
    Returns:
        PredictionResponse with:
        - predicted_class: Integer class ID (0-4)
        - label: Class label name
        - confidence: Confidence score for predicted class
        - probabilities: Dictionary with all class probabilities
        
    Raises:
        HTTPException: If image is invalid, empty, or model inference fails
    """
    global model_manager
    
    validate_model_ready()
    validate_upload(file)
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise ValueError("Uploaded file is empty")
        
        logger.info(f"Processing image: {file.filename} ({len(image_bytes)} bytes)")
        
        # Preprocess image
        input_tensor = preprocess_from_bytes(image_bytes)
        logger.info(f"Image preprocessed successfully, tensor shape: {input_tensor.shape}")
        
        # Run inference
        probabilities = model_manager.predict_proba(input_tensor)
        
        # Get predictions
        probs_np = probabilities.cpu().numpy()[0]  # Remove batch dimension
        predicted_class = int(probs_np.argmax())
        confidence = float(probs_np[predicted_class])
        
        # Create probability dictionary
        prob_dict = {i: float(probs_np[i]) for i in range(len(probs_np))}
        formatted_probs = format_probabilities(prob_dict)
        
        # Get class label
        label = get_label_from_class(predicted_class)
        
        logger.info(
            f"Prediction: class={predicted_class} ({label}), "
            f"confidence={confidence:.4f}"
        )
        
        return PredictionResponse(
            predicted_class=predicted_class,
            label=label,
            confidence=confidence,
            probabilities=formatted_probs
        )
        
    except ValueError as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


@app.post("/gradcam", tags=["Interpretability"])
async def gradcam(file: UploadFile = File(...), class_idx: Optional[int] = None):
    """
    Generate a Grad-CAM overlay for an uploaded fundus image.

    The heatmap is generated from the final convolutional layer in the
    MobileNetV2 branch of the hybrid model.
    """
    global model_manager

    validate_model_ready()
    validate_upload(file)

    if class_idx is not None and not 0 <= class_idx <= 4:
        raise HTTPException(
            status_code=400,
            detail="class_idx must be between 0 and 4"
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise ValueError("Uploaded file is empty")

        logger.info(f"Generating Grad-CAM for image: {file.filename} ({len(image_bytes)} bytes)")

        input_tensor, display_image = preprocess_image_and_array_from_bytes(image_bytes)
        overlay, predicted_class, confidence = generate_gradcam_overlay(
            model_manager=model_manager,
            input_tensor=input_tensor,
            display_image=display_image,
            class_idx=class_idx,
        )

        success, png_buffer = cv2.imencode(
            ".png",
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )
        if not success:
            raise RuntimeError("Failed to encode Grad-CAM image")

        label = get_label_from_class(predicted_class)
        return Response(
            content=png_buffer.tobytes(),
            media_type="image/png",
            headers={
                "X-Predicted-Class": str(predicted_class),
                "X-Predicted-Label": label,
                "X-Confidence": f"{confidence:.6f}",
                "X-Gradcam-Target": "Model_Mobile.features[-1]",
            },
        )

    except ValueError as e:
        logger.error(f"Grad-CAM preprocessing error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Grad-CAM error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during Grad-CAM generation: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled exceptions.
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
