from typing import Any, Dict

from pydantic.dataclasses import dataclass


@dataclass
class InferenceRequest:
    model_id: str
    """Model identifier"""
    inputs: Dict[str, Any]
    """Input data for inference"""
