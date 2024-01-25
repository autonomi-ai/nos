import sys

from rich import print

from nos.client import Client


GRPC_PORT = 50051
model_id = "tinyllama-1.1b-chat"


if __name__ == "__main__":
    client = Client(f"[::]:{GRPC_PORT}")
    assert client.WaitForServer()

    # Load the llama chat model
    model = client.Module(model_id)

    # Chat with the model
    query = "What is the meaning of life?"

    print()
    print("-" * 80)
    print(f">>> Chatting with the model (model={model_id}) ...")
    print(f"[bold yellow]Query: {query}[/bold yellow]")
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": query},
    ]
    for response in model.chat(messages=messages, max_new_tokens=1024, _stream=True):
        sys.stdout.write(response)
        sys.stdout.flush()
    print()
