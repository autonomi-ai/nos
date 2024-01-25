import subprocess

import openai


# Get the output of `sky status --ip nos-server` with subprocess
address = subprocess.check_output(["sky", "status", "--ip", "nos-server"]).decode("utf-8").strip()
print(f"Using address: {address}")

# Create a stream and print the output
client = openai.OpenAI(api_key="no-key-required", base_url=f"http://{address}:8000/v1")
stream = client.chat.completions.create(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    messages=[{"role": "user", "content": "Tell me a joke in 300 words"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
