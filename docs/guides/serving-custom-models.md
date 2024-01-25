In this guide, we will walk you through how to serve custom models with NOS. We will use the [WhisperX](https://github.com/m-bain/whisperX) model to build a custom runtime environment with docker, load the model and serve it via a gRPC/REST API. Feel free to navigate to [nos-playground/examples/whisperx](https://github.com/autonomi-ai/nos-playground/tree/main/examples/whisperx) for a full working example.

Here's the a short demo of the serving developer-experience:

[![asciicast](https://asciinema.org/a/618013.svg)](https://asciinema.org/a/618013?autoplay=1)

## 👩‍💻 Defining the custom model

The first step is to define the custom model in `models/whisperx.py`. Here we're using the popular [WhisperX](https://github.com/m-bain/whisperX) for transcribing audio files. Let's define a simple `WhisperX` class that wraps the `whisperx` package, loads the model and transcribes an audio file to a Python dictionary given its path.

```python linenums="1"  title="models/whisperx.py"
from pathlib import Path
from typing import Any, Dict, List

import torch

class WhisperX:
    def __init__(self, model_name: str = "large-v2"):
        import whisperx

        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        self.model = whisperx.load_model(model_name, self.device_str, compute_type="float16")
        self._load_align_model = whisperx.load_align_model
        self._align = whisperx.align

    def transcribe(
        self,
        path: Path,
        batch_size: int = 24,
        align_output: bool = True,
        language_code: str = "en",
    ) -> List[Dict[str, Any]]:
        """Transcribe the audio file."""
        with torch.inference_mode():
            result: Dict[str, Any] = self.model.transcribe(str(path), batch_size=batch_size)
            if align_output:
                alignment_model, alignment_metadata = self._load_align_model(
                    language_code=language_code, device=self.device_str
                )
                result = self._align(
                    result["segments"],
                    alignment_model,
                    alignment_metadata,
                    str(path),
                    self.device_str,
                    return_char_alignments=False,
                )
        return result
```

???info "Modular custom model registry"
    You will note that this file has nothing to do with NOS or it's dependencies. It is simply a regular Python class that wraps the `whisperx` package and loads the model. The `transcribe` method simply calls the `transcribe` method of the whisperx `model` and returns the result. This modularity is very much intentional as we would like to make sure that developers are not required to lock-in to any particular serving framework (i.e. ours), and instead focus on their specific modeling needs. 

## 📦 Defining the custom runtime environment

In the `models/whisperx.py` example shown above, we import `whisperx` which is a Python package that is not available in the default NOS runtime environment. To serve the model, we need to define a custom runtime environment that includes the `whisperx` package. We can do this by creating a custom docker runtime that installs the `whisperx` package and any other dependencies. 
o
With NOS, you simply define the custom runtime environment as part of the "images" key in a `serve.yaml` file. In the example below, we define a custom runtime environment called `whisperx-gpu` that is based on the `autonomi/nos:0.1.0-gpu` docker image. We then install the `whisperx` package and any other dependencies using the `pip` and `run` sub-commands.

!!!note
    You can look at the full list of `serve` CLI options [here](./cli/serve.md). The full `serve.yaml` specification is available [here](../cli/serve.md#serve-yaml-specification).

```yaml linenums="1" title="serve.yaml"
images:
  whisperx-gpu:
    base: autonomi/nos:0.1.0-gpu
    pip:
      - torchaudio>=2
      - faster-whisper>=0.8
      - pyannote.audio==3.0.1
      - transformers
      - ffmpeg-python>=0.2
      - pandas
      - setuptools>=65
      - nltk
    workdir: /app/whisperx
    run:
      - pip install --no-deps git+https://github.com/m-bain/whisperX.git

models:
  mbain-whisperx:
    model_cls: WhisperX
    model_path: models/whisperx.py
    default_method: transcribe
    runtime_env: whisperx-gpu
```

## 📦 Registering the custom whisperx model

The `serve.yaml` file also allows you to specify  the custom `mbain-whisperx` model that needs to be registered with NOS before you can serve it. The `model_cls` key specifies the class that we want to wrap and serve, and the `model_path` key specifies the corresponding path to the `whisperx.py` file. The `default_method: transcribe` key specifies the default method to call when the model is served. Finally, the `runtime_env: whisperx-gpu` key specifies the custom runtime docker environment that we defined above.

Via the `serve.yaml`, NOS automatically registers the new **WhisperX** model under a unique model-id, i.e. `mbain-whisperx` in this example. You can use the model-id to serve the model via a gRPC/REST API. For example, in order to use the client to call the `transcribe` method of the `WhisperX` model, we can simply do the following:

```python linenums="1" title="client.py"
from nos.client import Client

client = Client("[::]:50051")
```

!!!note
    While the `default_method` key allows you to specify a specific method to call, all methods of the class are also made available as callables through the exposed gRPC/REST API. For example, 


## 🚀 Serving the custom model

Now that we have defined the custom model and runtime environment, we can serve the model with NOS. To do this, we simply run the `nos serve` command and specify the `serve.yaml` file. 

```bash
nos serve up -c serve.yaml
```

Optionally, you can also start an HTTP gateway so that you can serve the model via a REST API. To do this, you can simply run the following command:

```bash
nos serve up -c serve.yaml --http
```

!!!note 
    Under the hood, `nos serve` builds a new custom runtime image based on the `whisperx-gpu` runtime environment we defined above. It then dyanmically registers the `WhsiperX` model class and serves it with the NOS inference server, exposing its methods via a gRPC/REST API. In this case, serving is done in a containerized environment, along-side the sidecar HTTP gateway (if specified).

## 📡 Using the custom model

Once the model is served, we can use the client to call the `transcribe` method of the `WhisperX` model. 

```python linenums="1" title="client.py"
from nos.client import Client

client = Client("[::]:50051")
client.WaitForServer()  # Wait for the server to start

model = client.Module("mbain-whisperx")
with client.UploadFile("test.wav") as remote_path:
    response = model.transcribe(path=remote_path)
    assert isinstance(response, dict)
    assert "segments" in response

for item in response["segments"]:
    assert "start" in item
    assert "end" in item
    assert "text" in item
```

In the example above, we use the client to call the `transcribe` method of the `WhisperX` model. We first upload the `test.wav` file to the server and then call the `transcribe` method with the remote path. The `transcribe` method returns a dictionary with the transcribed segments, just like the original `WhisperX` model, except that you have used the client-side API to have a remote server do the inference for you. 

!!!note 
    In this example, you could have also called `model(path=remote_path)` directly, since we registered `transcribe` as the `default_method` in the `serve.yaml` file.

That's it! You have successfully served a custom model with NOS.