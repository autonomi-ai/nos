from enum import Enum


class TaskType(Enum):
    """Task types."""

    OBJECT_DETECTION_2D = "object_detection_2d"
    """2D object detection."""
    IMAGE_SEGMENTATION_2D = "image_segmentation_2d"
    """2D image segmentation."""

    IMAGE_CLASSIFICATION = "image_classification"
    """Image classification."""
    IMAGE_GENERATION = "image_generation"
    """Image generation."""

    IMAGE_EMBEDDING = "image_embedding"
    """Image embedding."""
    TEXT_EMBEDDING = "text_embedding"
    """Text embedding."""
