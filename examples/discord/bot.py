import asyncio
import io
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import discord
from discord.ext import commands
from diskcache import Cache

from nos.client import InferenceClient, TaskType
from nos.constants import NOS_TMP_DIR
from nos.logging import logger


@dataclass
class LoRAPromptModel:

    thread_id: str
    """Discord thread ID"""

    thread_name: str
    """Discord thread name"""

    model_id: str
    """Training job ID / model ID"""

    prompt: str
    """Prompt used to train the model"""

    @property
    def job_id(self) -> str:
        return self.model_id

    def __str__(self) -> str:
        return f"LoRAPromptModel(thread_id={self.thread_id}, thread_name={self.thread_name}, model_id={self.model_id}, prompt={self.prompt})"


NOS_PLAYGROUND_CHANNEL = "nos-playground"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
# BASE_MODEL = "stabilityai/stable-diffusion-2-1"

# Init nos server, wait for it to spin up then confirm its healthy:
client = InferenceClient()

logger.debug("Waiting for server to start...")
client.WaitForServer()

logger.debug("Confirming server is healthy...")
if not client.IsHealthy():
    raise RuntimeError("NOS server is not healthy")

logger.debug("Server is healthy!")
NOS_VOLUME_DIR = Path(client.Volume())
NOS_TRAINING_VOLUME_DIR = Path(client.Volume("nos-playground"))
logger.debug(f"Creating training data volume [volume={NOS_TRAINING_VOLUME_DIR}]")

# Set permissions for our bot to allow it to read messages:
intents = discord.Intents.default()
intents.message_content = True

# Create our bot, with the command prefix set to "/":
bot = commands.Bot(command_prefix="/", intents=intents)

logger.debug("Starting bot, initializing existing threads ...")

# Maps channel_id -> LoRAPromptModel
MODEL_DB = Cache(str(NOS_TMP_DIR / NOS_PLAYGROUND_CHANNEL))


@bot.command()
async def generate(ctx, *, prompt):
    """Create a callback to read messages and generate images from prompt"""
    logger.debug(
        f"/generate [prompt={prompt}, channel={ctx.channel.name}, channel_id={ctx.channel.id}, user={ctx.author.name}]"
    )

    st = time.perf_counter()
    if ctx.channel.name == NOS_PLAYGROUND_CHANNEL:
        # Pull the channel id so we know which model to run:

        # Acknowledge the request, by reacting to the message with a checkmark
        logger.debug(f"Request acknowledged [id={ctx.message.id}]")
        await ctx.message.add_reaction("✅")

        logger.debug(f"/generate request submitted [id={ctx.message.id}]")
        response = client.Run(
            task=TaskType.IMAGE_GENERATION,
            model_name=BASE_MODEL,
            prompts=[prompt],
            width=512,
            height=512,
            num_images=1,
        )
        logger.debug(f"/generate request completed [id={ctx.message.id}, elapsed={time.perf_counter() - st:.2f}s]")
        (image,) = response["images"]
        thread = ctx
    else:
        # Pull the channel id so we know which model to run:
        thread = ctx.channel
        thread_id = ctx.channel.id
        model = MODEL_DB.get(thread_id, default=None)
        if model is None:
            logger.debug(f"Failed to fetch model [thread_id={thread_id}]")
            await thread.send("No model found")
            return

        # Get model info
        try:
            models = client.ListModels()
            info = {m.name: m for m in models}[model.model_id]
            logger.debug(f"Got model info [model_id={model.model_id}, info={info}]")
        except Exception as e:
            logger.debug(f"Failed to fetch model info [model_id={model.model_id}, e={e}]")
            await thread.send(f"Failed to fetch model [model_id={model.model_id}]")
            return

        # Acknowledge the request, by reacting to the message with a checkmark
        logger.debug(f"Request acknowledged [id={ctx.message.id}]")
        await ctx.message.add_reaction("✅")

        # Run inference on the trained model
        logger.debug(f"/generate request submitted [id={ctx.message.id}, model_id={model.model_id}, model={model}]")
        response = client.Run(
            task=TaskType.IMAGE_GENERATION,
            model_name=model.model_id,
            prompts=[prompt],
            width=512,
            height=512,
            num_images=1,
        )
        logger.debug(
            f"/generate request completed [id={ctx.message.id}, model={model}, elapsed={time.perf_counter() - st:.2f}s]"
        )
        (image,) = response["images"]

    # Save the image to a buffer and send it back to the user:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    await thread.send(f"{prompt}", file=discord.File(image_bytes, filename=f"{ctx.message.id}.png"))


@bot.command()
async def train(ctx, *, prompt):
    logger.debug(f"/train [channel={ctx.channel.name}, channel_id={ctx.channel.id}, user={ctx.author.name}]")

    if ctx.channel.name != NOS_PLAYGROUND_CHANNEL:
        logger.debug("ignoring [channel={ctx.channel.name}]")
        return

    if not ctx.message.attachments:
        logger.debug("no attachments to train on, returning!")
        return

    if "sks" not in prompt:
        await ctx.send("Please include 'sks' in your training prompt!")
        return

    # Create a thread for this training job
    message_id = str(ctx.message.id)
    thread = await ctx.message.create_thread(name=f"{prompt} ({message_id})")
    logger.debug(f"Created thread [id={thread.id}, name={thread.name}]")

    # Save the thread id
    thread_id = thread.id

    # Create the training directory for this thread
    dirname = NOS_TRAINING_VOLUME_DIR / str(thread_id)
    dirname.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created training directory [dirname={dirname}]")

    # Save all attachments to the training directory
    logger.debug(f"Saving attachments [dirname={dirname}, attachments={len(ctx.message.attachments)}]")
    for attachment in ctx.message.attachments:
        filename = str(dirname / f"{str(uuid.uuid4().hex[:8])}_{attachment.filename}")
        await attachment.save(filename)
        logger.debug(f"Saved attachment [filename={filename}]")

    # Acknowledge the request, by reacting to the message with a checkmark
    logger.debug(f"Request acknowledged [id={ctx.message.id}]")
    await ctx.message.add_reaction("✅")

    # Train a new LoRA model with the image of a bench
    response = client.Train(
        method="stable-diffusion-dreambooth-lora",
        inputs={
            "model_name": BASE_MODEL,
            "instance_directory": dirname.relative_to(NOS_VOLUME_DIR),
            "instance_prompt": prompt,
            "max_train_steps": 500,
        },
        metadata={
            "name": "sdv15-dreambooth-lora",
        },
    )
    job_id = response["job_id"]
    logger.debug(f"Submitted training job [id={thread_id}, response={response}, dirname={dirname}]")
    await thread.send(f"@here Submitted training job [id={thread.name}, model={job_id}]")

    # Save the model
    MODEL_DB[thread_id] = LoRAPromptModel(
        thread_id=thread_id,
        thread_name=thread.name,
        model_id=f"custom/{job_id}",
        prompt=prompt,
    )
    logger.debug(f"Saved model [id={thread_id}, model={MODEL_DB[thread_id]}]")

    if response is None:
        logger.error(f"Failed to submit training job [id={thread_id}, response={response}, dirname={dirname}]")
        await thread.send(f"Failed to train [prompt={prompt}, response={response}, dirname={dirname}]")

    # Create a new thread to watch the training job
    async def post_on_training_complete_async():
        # Wait for the model to be ready
        response = client.Wait(job_id=job_id, timeout=600, retry_interval=10)
        logger.debug(f"Training completed [job_id={job_id}, response={response}].")

        # Get the thread
        _thread = bot.get_channel(thread_id)
        await _thread.send(f"@here Training complete [id={_thread.name}, model={job_id}]")

        # Wait for model to be registered after the job is complete
        await asyncio.sleep(5)

        # Run inference on the trained model
        st = time.perf_counter()
        response = client.Run(
            task=TaskType.IMAGE_GENERATION,
            model_name=f"custom/{job_id}",
            prompts=[prompt],
            width=512,
            height=512,
            num_images=1,
        )
        logger.debug(f"/generate request completed [model={job_id}, elapsed={time.perf_counter() - st:.2f}s]")
        (image,) = response["images"]

        # Save the image to a buffer and send it back to the user:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        await _thread.send(f"{prompt}", file=discord.File(image_bytes, filename=f"{ctx.message.id}.png"))

    # def post_on_training_complete():
    #     asyncio.run(post_on_training_complete_async())

    logger.debug(f"Starting thread to watch training job [id={thread_id}, job_id={job_id}]")
    # threading.Thread(target=post_on_training_complete, daemon=True).start()
    asyncio.run_coroutine_threadsafe(post_on_training_complete_async(), loop)
    logger.debug(f"Started thread to watch training job [id={thread_id}, job_id={job_id}]")


# Pull API token out of environment and run the bot:
bot_token = os.environ.get("DISCORD_BOT_TOKEN")
if bot_token is None:
    raise Exception("DISCORD_BOT_TOKEN environment variable not set")
logger.debug(f"Starting bot with token [token={bot_token[:5]}****]")
# bot.loop.run_until_complete(setup())


async def run_bot():
    await bot.start(bot_token)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(run_bot())
    loop.run_forever()
