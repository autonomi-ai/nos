"""Llama2 Chat model tests and benchmarks."""
import pytest

from nos.models import Llama2Chat
from nos.test.utils import skip_if_no_torch_cuda


SYSTEM_PROMPT = "You are NOS chat, a Llama 2 large language model (LLM) agent hosted by Autonomi AI."


@pytest.fixture(scope="module")
def model():
    MODEL_NAME = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
    yield Llama2Chat(model_name=MODEL_NAME)


@skip_if_no_torch_cuda
def test_llama2_chat(model):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is the meaning of life?"},
    ]
    for _ in model.chat(messages=messages, max_new_tokens=512):
        pass
