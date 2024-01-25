import pytest

from nos.common.helpers import memory_bytes


def test_memory_bytes_MB():
    result = memory_bytes("512MB")
    assert result == 512 * 1000**2


def test_memory_bytes_M():
    result = memory_bytes("256M")
    assert result == 256 * 1000**2
    result = memory_bytes("256.5M")
    assert result == 256_500_000


def test_memory_bytes_Mi():
    result = memory_bytes("128Mi")
    assert result == 128 * 1024**2
    result = memory_bytes("128.5Mi")
    assert result == int(128.5 * 1024**2)


def test_memory_bytes_GB():
    result = memory_bytes("2GB")
    assert result == 2 * 1000**3
    result = memory_bytes("2.5GB")
    assert result == 2_500_000_000


def test_memory_bytes_G():
    result = memory_bytes("1G")
    assert result == 1 * 1000**3
    result = memory_bytes("1.5G")
    assert result == 1_500_000_000


def test_memory_bytes_Gi():
    result = memory_bytes("4Gi")
    assert result == 4 * 1024**3
    result = memory_bytes("4.5Gi")
    assert result == int(4.5 * 1024**3)


def test_memory_bytes_numeric():
    result = memory_bytes("1024")
    assert result == 1024

    result = memory_bytes("1_000_000")
    assert result == 1_000_000


def test_memory_bytes_invalid():
    with pytest.raises(ValueError):
        memory_bytes("invalid")
