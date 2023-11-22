import datetime
import uuid
from typing import List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field


class ChatModel(BaseModel):
    id: str
    """ID of the model"""

    object: Literal["model"] = Field(default="model")
    """Type of the model"""

    created: int = Field(default_factory=lambda: int(datetime.datetime.utcnow().timestamp()))
    """UNIX timestamp (in seconds) of when the model was created"""

    owned_by: str = Field(default="autonomi-ai")
    """ID of the organization that owns the model"""


TModel = TypeVar("TModel", bound="Model")


class Model(BaseModel):
    data: List[ChatModel]
    """List of models"""
    object: str = Field(default="list")
    """Type of the list"""

    @classmethod
    def list(cls) -> TModel:
        pass


class Message(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(default="user")
    """Role of the message, either 'user', 'assistant' or 'system'"""
    content: Optional[str] = None
    """Content of the message"""


class Choice(BaseModel):
    finish_reason: Optional[Literal["stop"]] = Field(examples="stop")
    """Reason for finishing the conversation"""

    message: Optional[Message] = None
    """Message to add to the conversation"""

    index: int = 0
    """Index of the choice"""


class Usage(BaseModel):
    completion_tokens: int
    """Number of tokens used for the completion"""

    prompt_tokens: int
    """Number of tokens used for the prompt"""

    total_tokens: int
    """Total number of tokens used (prompt + completion)"""


class ChatCompletionsRequest(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")  # noqa: A003
    """ID of the completion"""

    messages: List[Message]
    """Messages to complete"""

    max_tokens: int = 250
    """Maximum number of tokens to generate"""

    temperature: float = 0.7
    """Temperature of the sampling distribution"""

    top_p: float = 1.0
    """Cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling"""

    top_k: Optional[int] = None
    """Number of highest probability vocabulary tokens to keep for top-k-filtering"""

    logprobs: Optional[int] = None
    """Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens"""

    model: str = "meta-llama/Llama-2-7b-chat-hf"
    """Model to use for completion"""

    stream: bool = False
    """Whether to stream the response or not"""


class Completion(BaseModel):
    """https://platform.openai.com/docs/api-reference/chat/object"""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")  # noqa: A003
    """ID of the completion"""

    object: Literal["chat.completion"] = Field(default="chat.completion")
    """Type of the completion"""

    created: int = Field(default_factory=lambda: int(datetime.datetime.utcnow().timestamp()))
    """UNIX timestamp (in seconds of when the chat completion was created"""

    model: str
    """Model used for the chat completion"""

    choices: List[Choice]
    """Choices made during the chat completion"""

    usage: Optional[Usage] = None
    """Usage information for the chat completion"""
