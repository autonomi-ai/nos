Whispernet audio transcription with serialized Audio Files. The current interface requires that a `.wav` be serialized into base64 and passed in as a string. The following snippet demonstrates a simple youtube->text transcription flow with `ytd`.

```python
from nos.client import Client
from nos.common import TaskType
from yt_dlp import YoutubeDL

nos.init(runtime="gpu")
client = Client()
client.WaitForServer()
client.IsHealthy()

# Short util to extract .wav files from youtube urls
def download_youtube_url_and_transcribe(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        # set download location to current directory
        info_dict = ydl.extract_info(url, download=False)
        output_filename = ydl.prepare_filename(info_dict)
        audio_filename = output_filename.replace(".webm", ".wav")
        error_code = ydl.download([url])
        assert error_code == 0

    with open(audio_filename, "rb") as f:
        audio_data = f.read()

    # serialize wav to base64
    import base64
    audio_data_base64 = base64.b64encode(audio_data).decode("utf-8")

    # run transcription
    predictions = client.Run("openai/whisper-tiny.en", inputs={"audio" : audio_data_base64})
    print(predictions["text"])

youtube_url = "https://www.youtube.com/watch?v=EFfJEB1jkSo"
download_youtube_url_and_transcribe(youtube_url)
```
