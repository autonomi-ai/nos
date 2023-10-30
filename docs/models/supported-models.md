This README lists the models supported by NOS, along with their corresponding links to Hugging Face or Torch Hub, and the supported devices (CPU or GPU). Navigate to our [models](https://github.com/autonomi-ai/nos/tree/main/nos/models) page for more up-to-date information.

<table>
    <tr>
        <td><b>Modality</b></td>
        <td><b>Task</b></td>
        <td><b>Model Name</b></td>
        <td><b>Supported Devices</b></td>
        <td><b><center>API</center></b></td>
    </tr>

    <tr>
        <td>🏞️</td>
        <td><b>Object Detection</b></td>
        <td><a href="https://github.com/Megvii-BaseDetection/YOLOX">YOLOX</a></td>
        <td>CPU, GPU</td>
        <td>

            ```python
            img = Image.open("test.png")

            yolox = client.Module("yolox/nano")
            predictions = yolox(images=img)
            # {"bboxes": ..., "scores": ..., "labels": ...}
            ```
        </td>
    </tr>

    <tr>
        <td>🏞️</td>
        <td><b>Depth Estimation</b></td>
        <td><a href="https://github.com/isl-org/MiDaS">MiDaS</a></td>
        <td>CPU, GPU</td>
        <td>

            ```python
            img = Image.open("test.png")

            model = client.Module("isl-org/MiDaS")
            result = model(images=img)
            # {"depths": np.ndarray}
            ```
        </td>
    </tr>

    <tr>
        <td>📝, 🏞️</td>
        <td><b>Text-Image Embedding</b></td>
        <td><a href="https://huggingface.co/openai/clip-vit-base-patch32">OpenAI - CLIP</a></td>
        <td>CPU, GPU</td>
        <td>
            ```python
            img = Image.open("test.png")

            clip = client.Module("openai/clip-vit-base-patch32")
            img_vec = clip.encode_image(images=img)
            txt_vec = clip.encode_text(text=["fox jumped over the moon"])
            ```
        </td>
    </tr>

    <tr>
        <td>📝, 🏞️</td>
        <td><b>Text/Input Conditioned Image Segmentation</b></td>
        <td><a href="https://huggingface.co/facebook/sam-vit-large">Facebook Research - Segment Anything</a></td>
        <td>CPU, GPU</td>
        <td>
            ```python
            img = Image.open("test.png")

            model = client.Module("facebook/sam-vit-large")
            outputs: List[np.ndarray] = model(images=img, grid_size=20)
            ```
        </td>
    </tr>

    <tr>
        <td>📝, 🏞️</td>
        <td><b>Text-to-Image Generation</b></td>
        <td><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">Stability AI - Stable Diffusion XL</a></td>
        <td>GPU</td>
        <td>
            ```python
            sdxl = client.Module("stabilityai/stable-diffusion-xl-base-1-0")
            sdxl(prompts=["fox jumped over the moon"],
                 width=1024, height=1024, num_images=1)
            ```
        </td>
    </tr>

    <tr>
        <td>📝, 🏞️</td>
        <td><b>Text-to-Image Generation</b></td>
        <td><a href="https://huggingface.co/stabilityai/stable-diffusion-2-1">Stability AI - Stable Diffusion 2.1</a></td>
        <td>GPU</td>
        <td>
            ```python
            sdv2 = client.Module("stabilityai/stable-diffusion-2-1")
            sdv2(prompts=["fox jumped over the moon"],
                 width=512, height=512, num_images=1)
            ```
        </td>
    </tr>

    <tr>
        <td>📝, 🏞️</td>
        <td><b>Text-to-Image Generation</b></td>
        <td><a href="https://huggingface.co/stabilityai/stable-diffusion-2">Stability AI - Stable Diffusion 2</a></td>
        <td>GPU</td>
        <td>
            ```python
            sdv2 = client.Module("stabilityai/stable-diffusion-2")
            sdv2(prompts=["fox jumped over the moon"],
                 width=512, height=512, num_images=1)
            ```
        </td>
    </tr>

    <tr>
        <td>📝, 🏞️</td>
        <td><b>Text-to-Image Generation</b></td>
        <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">RunwayML - Stable Diffusion v1.5</a></td>
        <td>CPU, GPU</td>
        <td>
            ```python
            sdv2 = client.Module("runwayml/stable-diffusion-v1-5")
            sdv2(prompts=["fox jumped over the moon"],
                 width=512, height=512, num_images=1)
            ```
        </td>
    </tr>

    <tr>
        <td>🎙️</td>
        <td><b>Speech-to-Text</b></td>
        <td><a href="https://huggingface.co/openai/whisper-large-v2">OpenAI - Whisper</a></td>
        <td>GPU</td>
        <td>
            ```python
            from base64 import b64encode

            whisper = client.Module("openai/whisper-large-v2")
            with open("test.wav", "rb") as f:
                audio_data = f.read()
                audio_b64 = b64encode(audio_data).decode("utf-8")
                transcription = whisper.transcribe(audio=audio_64)
            ```
        </td>
    </tr>

    <tr>
        <td>🎙️</td>
        <td><b>Text-to-Speech</b></td>
        <td><a href="https://huggingface.co/suno/bark">Suno - Bark</a></td>
        <td>GPU</td>
        <td>
            ```python
            bark = client.Module("suno/bark")
            audio_data = bark(prompts=["fox jumped over the moon"])
            ```
        </td>
    </tr>
</table>
