def memory_bytes(memory: str) -> int:
    """Convert a human-readable memory string to bytes."""
    if memory.endswith("KB") or memory.endswith("K"):
        memory = memory.split("K")[0]
        memory = int(float(memory) * 1000)
    elif memory.endswith("Ki"):
        memory = memory.split("K")[0]
        memory = int(float(memory) * 1024)
    if memory.endswith("MB") or memory.endswith("M"):
        memory = memory.split("M")[0]
        memory = int(float(memory) * 1000**2)
    elif memory.endswith("Mi"):
        memory = memory.split("M")[0]
        memory = int(float(memory) * 1024**2)
    elif memory.endswith("GB") or memory.endswith("G"):
        memory = memory.split("G")[0]
        memory = int(float(memory) * 1000**3)
    elif memory.endswith("Gi"):
        memory = memory.split("G")[0]
        memory = int(float(memory) * 1024**3)
    else:
        try:
            memory = int(memory)
        except ValueError:
            raise ValueError(f"Could not parse memory string {memory}")
    return memory
