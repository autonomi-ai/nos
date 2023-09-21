from typing import Any, Dict, Optional

from fastapi import File, UploadFile
from pydantic import BaseModel


class InferenceRequest(BaseModel):
    task: str
    """Task used for inference"""
    model_name: str
    """Model identifier"""
    inputs: Dict[str, Any]
    """Input data for inference"""
    data: Optional[UploadFile] = File(None)
    """Uploaded image / video / audio file for inference"""
