import gradio as gr

from nos.client import InferenceClient, TaskType
from nos.logging import logger

# Init NOS server, wait for it to spin up then confirm its healthy.
client = InferenceClient()
logger.debug("Waiting for server to start...")
client.WaitForServer()

logger.debug("Confirming server is healthy...")
if not client.IsHealthy():
    raise RuntimeError("NOS server is not healthy")

# Setup to load youtube videos from url, extract audio and record text.
import youtube_dl
from moviepy.editor import AudioFileClip
def download_youtube_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_url = info_dict.get("url", None)
        audio = AudioFileClip(video_url)
        return audio
    
# Transcribe audio using WhisperNet
def transcribe_audio(audio):
    # Convert audio to mono and resample to 16000 Hz (required for WhisperNet)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    # Load the pre-trained WhisperNet model
    model = torch.hub.load('snakers4/silero-models', 'silero-whisper-large')

    # Perform ASR (Automatic Speech Recognition)
    waveform, sample_rate = audio.to_soundarray(), 16000
    with torch.no_grad():
        transcript = model(waveform)

    return transcript

# Create the Gradio interface
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.inputs.Textbox(label="YouTube URL"),
    outputs=gr.outputs.Textbox(label="Transcription"),
    live=True
).launch()

"""
# Define the function that will run the model
def run_stable_diffusion(prompt):
    response = client.Run(
        task=TaskType.IMAGE_GENERATION,
        model_name="runwayml/stable-diffusion-v1-5",
        prompts=[prompt],
        width=512,
        height=512,
        num_images=1,
    )
    (image,) = response["images"]

    return image 

# Create a Gradio interface
iface = gr.Interface(
    fn=run_stable_diffusion,
    inputs="text",  # You can specify input types like "image", "text", etc.
    outputs="image", # You can specify output types like "image", "text", etc.
    title="Stable Diffusion",
    description="Image generation with NOS.",
)
"""

# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch()
