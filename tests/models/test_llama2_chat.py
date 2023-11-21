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
    from nos.common import tqdm

    for _ in tqdm(
        model.chat(message="What is the meaning of life?", system_prompt=SYSTEM_PROMPT, max_new_tokens=512), skip=1
    ):
        pass
