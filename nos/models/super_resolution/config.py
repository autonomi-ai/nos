from dataclasses import dataclass

from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class SuperResolutionConfig(HuggingFaceHubConfig):
    """SuperResolution model configuration."""

    method: str = "ldm"
    """SuperResolution method (choice of `ldm` or `swin2sr`)."""

    def __post_init__(self):
        if self.method not in ("ldm", "swin2sr"):
            raise ValueError(f"Invalid method: {self.method}, available methods: ['ldm', 'swin2sr']")
