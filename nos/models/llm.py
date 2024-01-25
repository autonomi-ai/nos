import time
from dataclasses import dataclass
from threading import Thread
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from nos import hub
from nos.common import TaskType
from nos.hub import HuggingFaceHubConfig, hf_login


SYSTEM_PROMPT = "You are NOS chat, a Llama 2 large language model (LLM) agent hosted by Autonomi AI."

# Note (spillai): This default llama2 chat template removes
# the error when consecutive user messages are provided.
LLAMA2_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\\n\\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'t know the answer to a question, please don\\'t share false information.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"


@dataclass(frozen=True)
class LLMConfig(HuggingFaceHubConfig):
    """Llama2 chat model configuration."""

    max_new_tokens: int = 2048
    """Maximum number of tokens to generate."""

    max_input_token_length: int = 4096
    """Maximum number of tokens in the input."""

    compute_dtype: str = "float16"
    """Compute type for the model."""

    needs_auth: bool = False
    """Whether the model needs authentication."""

    additional_kwargs: Dict[str, Any] = None
    """Additional keyword arguments to pass to the model."""

    chat_template: str = None
    """Chat template to use for the model."""


class LLM:
    configs = {
        "meta-llama/Llama-2-7b-chat-hf": LLMConfig(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            compute_dtype="float16",
            needs_auth=True,
            chat_template=LLAMA2_CHAT_TEMPLATE,
        ),
        "meta-llama/Llama-2-13b-chat-hf": LLMConfig(
            model_name="meta-llama/Llama-2-13b-chat-hf",
            compute_dtype="float16",
            needs_auth=True,
            chat_template=LLAMA2_CHAT_TEMPLATE,
        ),
        "meta-llama/Llama-2-70b-chat-hf": LLMConfig(
            model_name="meta-llama/Llama-2-70b-chat-hf",
            compute_dtype="float16",
            needs_auth=True,
            chat_template=LLAMA2_CHAT_TEMPLATE,
        ),
        "HuggingFaceH4/zephyr-7b-beta": LLMConfig(
            model_name="HuggingFaceH4/zephyr-7b-beta",
            compute_dtype="float16",
        ),
        "HuggingFaceH4/tiny-random-LlamaForCausalLM": LLMConfig(
            model_name="HuggingFaceH4/tiny-random-LlamaForCausalLM",
            compute_dtype="float16",
        ),
        "NousResearch/Yarn-Mistral-7b-128k": LLMConfig(
            model_name="NousResearch/Yarn-Mistral-7b-128k",
            compute_dtype="float16",
            additional_kwargs={"use_flashattention_2": True, "trust_remote_code": True},
        ),
        "mistralai/Mistral-7B-Instruct-v0.2": LLMConfig(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            compute_dtype="float16",
        ),
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": LLMConfig(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            compute_dtype="float16",
        ),
    }

    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        from nos.logging import logger

        try:
            self.cfg = LLM.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {LLMConfig.configs.keys()}")

        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)

        token = hf_login() if self.cfg.needs_auth else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=getattr(torch, self.cfg.compute_dtype),
            token=token,
            device_map=self.device_str,
            **(self.cfg.additional_kwargs or {}),
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, token=token)
        self.tokenizer.use_default_system_prompt = False
        self.logger = logger

    @torch.inference_mode()
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        num_beams: int = 1,
    ) -> Iterable[str]:
        """Chat with the model."""
        self.logger.debug(f"Conversation: {messages}")
        input_ids = self.tokenizer.apply_chat_template(
            messages, chat_template=self.cfg.chat_template, return_tensors="pt"
        )
        if input_ids.shape[1] > self.cfg.max_input_token_length:
            input_ids = input_ids[:, -self.cfg.max_input_token_length :]
            self.logger.warning(
                f"Trimmed input from conversation as it was longer than {self.cfg.max_input_token_length} tokens."
            )
        input_ids = input_ids.to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, timeout=180.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            {"input_ids": input_ids},
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            use_cache=True,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        start_t = None
        for idx, text in enumerate(streamer):
            yield text
            # We only measure the time after the first token is generated
            if start_t is None:
                start_t = time.perf_counter()
            if idx > 0:
                self.logger.debug(
                    f"""tok/s={idx / (time.perf_counter() - start_t):.1f}, """
                    f"""memory={torch.cuda.memory_allocated(device=self.model.device) / 1024 ** 2:.1f} MB, """
                    f"""allocated={torch.cuda.max_memory_allocated(device=self.model.device) / 1024 ** 2:.1f} MB, """
                    f"""peak={torch.cuda.max_memory_reserved(device=self.model.device) / 1024 ** 2:.1f} MB, """
                )


for model_name in LLM.configs:
    cfg = LLM.configs[model_name]
    hub.register(
        model_name,
        TaskType.TEXT_GENERATION,
        LLM,
        init_args=(model_name,),
        method="chat",
    )
