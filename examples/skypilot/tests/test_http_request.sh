curl \
    -X POST http://$(sky status --ip nos-server):8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": "Tell me a joke in 300 words"}],
        "temperature": 0.7, "stream": true
      }'
