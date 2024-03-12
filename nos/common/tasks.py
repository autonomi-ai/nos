from enum import Enum


class TaskType(str, Enum):
    """Task types."""

    OBJECT_DETECTION_2D = "object_detection_2d"
    """2D object detection."""
    IMAGE_SEGMENTATION_2D = "image_segmentation_2d"
    """2D image segmentation."""
    DEPTH_ESTIMATION_2D = "depth_estimation_2d"
    """2D depth estimation."""

    IMAGE_CLASSIFICATION = "image_classification"
    """Image classification."""
    IMAGE_GENERATION = "image_generation"
    """Image generation."""
    IMAGE_SUPER_RESOLUTION = "image_super_resolution"
    """Image super-resolution."""

    IMAGE_EMBEDDING = "image_embedding"
    """Image embedding."""
    TEXT_EMBEDDING = "text_embedding"
    """Text embedding."""

    TEXT_GENERATION = "text_generation"
    AUDIO_TRANSCRIPTION = "audio_transcription"

    CUSTOM = "custom"
    """Custom task type."""
