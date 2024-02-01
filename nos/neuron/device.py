import os
from dataclasses import dataclass

from nos.constants import NOS_CACHE_DIR
from nos.logging import logger


@dataclass
class NeuronDevice:
    """Neuron device environment."""

    _instance: "NeuronDevice" = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def device_count() -> int:
        import torch_neuronx

        try:
            return torch_neuronx.xla_impl.data_parallel.device_count()
        except (RuntimeError, AssertionError):
            return 0

    @staticmethod
    def setup_environment() -> None:
        """Setup neuron environment."""
        for k, v in os.environ.items():
            if "NEURON" in k:
                logger.debug(f"{k}={v}")
        cores: int = int(os.getenv("NOS_NEURON_CORES", 2))
        logger.info(f"Setting up neuron env with {cores} cores")
        cache_dir = NOS_CACHE_DIR / "neuron"
        os.environ["NEURONX_CACHE"] = "on"
        os.environ["NEURONX_DUMP_TO"] = str(cache_dir)
        os.environ["NEURON_RT_NUM_CORES"] = str(cores)
        os.environ["NEURON_RT_VISIBLE_CORES"] = ",".join([str(i) for i in range(cores)])
        os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer-inference"
