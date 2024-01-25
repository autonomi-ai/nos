import sys
from pathlib import Path

import rich.console

from nos.client import Client


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize audio file.")
    parser.add_argument("--filename", type=str, help="Audio file to summarize.")
    args = parser.parse_args()

    console = rich.console.Console()
    path = Path(args.filename)

    # Create a client
    address = "[::]:50051"
    print(f"Connecting to client at {address} ...")
    client = Client(address)
    client.WaitForServer()

    # Transcribe with Whisper
    model_id = "distil-whisper/distil-small.en"
    model = client.Module(model_id)
    console.print()
    console.print(f"[bold white]Transcribe with [yellow]{model_id}[/yellow].[/bold white]")

    # Transcribe the audio file and print the text
    transcription_text = ""
    print(f"Transcribing audio file: {path}")
    with client.UploadFile(path) as remote_path:
        response = model.transcribe(path=remote_path, batch_size=8)
        for item in response["chunks"]:
            transcription_text += item["text"]
            sys.stdout.write(item["text"])
            sys.stdout.flush()
    print()

    # Summarize the transcription with LLMs
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llm = client.Module(model_id)
    console.print()
    console.print("[bold white]Summarize with [yellow]TinyLlama/TinyLlama-1.1B-Chat-v1.0[/yellow].[/bold white]")

    prompt = f"""
    You are a useful transcribing assistant.
    Summarize the following text concisely with key points.
    Keep the sentences short, highlight key concepts in each bullet starting with a hyphen.

    {transcription_text}
    """
    messages = [
        {"role": "user", "content": prompt},
    ]
    for response in llm.chat(messages=messages, max_new_tokens=1024, _stream=True):
        sys.stdout.write(response)
        sys.stdout.flush()
