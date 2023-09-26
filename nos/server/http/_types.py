from typing import Any, Dict

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    task: str
    """Task used for inference"""
    model_name: str
    """Model identifier"""
    inputs: Dict[str, Any]
    """Input data for inference"""
