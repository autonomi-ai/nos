import pytest

from nos.common.runtime import is_torch_neuron_available


pytestmark = pytest.mark.skipif(not is_torch_neuron_available(), reason="Requires torch_neuron")


def test_neuron_device():
    from nos.neuron.device import NeuronDevice

    neuron_env = NeuronDevice.get()
    assert neuron_env.device_count() > 0
    assert neuron_env.setup_environment() is None
