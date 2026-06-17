"""
Utility functions for the API.
"""

# Class mapping for Diabetic Retinopathy classification
CLASS_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

def get_label_from_class(class_id: int) -> str:
    """
    Get the label name for a given class ID.
    
    Args:
        class_id: Integer class ID (0-4)
        
    Returns:
        Label name as string
    """
    return CLASS_LABELS.get(class_id, "Unknown")


def format_probabilities(probs: dict) -> dict:
    """
    Format probabilities with proper labels and rounded values.
    
    Args:
        probs: Dictionary with class IDs as keys and probabilities as values
        
    Returns:
        Dictionary with label names as keys and probabilities as values
    """
    return {
        get_label_from_class(class_id): round(prob, 4)
        for class_id, prob in probs.items()
    }
