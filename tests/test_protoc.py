from pathlib import Path


def test_dynamic_protoc():
    """Test dynamic protoc."""
    from nos.protoc import DynamicProtobufCompiler, import_module

    compiler = DynamicProtobufCompiler.get()
    assert compiler is not None

    assert Path(DynamicProtobufCompiler.cache_dir).exists()
    nos_service_pb2 = import_module("nos_service_pb2")
    assert nos_service_pb2 is not None
    nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")
    assert nos_service_pb2_grpc is not None

    compiled_artifacts = list(DynamicProtobufCompiler.cache_dir.glob("*.py"))
    assert len(compiled_artifacts) > 0
