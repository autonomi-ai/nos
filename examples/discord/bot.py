import io
import os
import time
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


NOS_PLAYGROUND_CHANNEL = "nos-playground"

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
        logger.debug(f"/generate request submitted [id={ctx.message.id}]")
        response = client.Run(
            task=TaskType.IMAGE_GENERATION,
            model_name="stabilityai/stable-diffusion-2",
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
            logger.debug(f"Failed to fetch model [thread_id={thread_id}")
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

    if "<sks>" not in prompt:
        await ctx.send("Please include a <sks> in your training prompt!")
        return

    # Create a thread for this training job
    message_id = str(ctx.message.id)
    thread = await ctx.channel.create_thread(name=f"{prompt} ({message_id})", type=discord.ChannelType.public_thread)
    await thread.send(f"Created a new thread for training [id={thread.name}]")

    # Save the thread id
    thread_id = thread.id

    # Create the training directory for this thread
    dirname = NOS_TRAINING_VOLUME_DIR / str(thread_id)
    dirname.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created training directory [dirname={dirname}]")

    # Save all attachments to the training directory
    for attachment in ctx.message.attachments:
        logger.debug(f"Got attachement [filename={attachment.filename}]")
        await attachment.save(str(dirname / str(attachment.filename)))
        logger.debug(f"Saved attachment [filename={attachment.filename}]")

    # Train a new LoRA model with the image of a bench
    response = client.Train(
        method="stable-diffusion-dreambooth-lora",
        inputs={
            "model_name": "stabilityai/stable-diffusion-2-1",
            "instance_directory": dirname.relative_to(NOS_VOLUME_DIR),
            "instance_prompt": "A photo of a <sks> on the moon",
            "max_train_steps": 10,
        },
        metadata={
            "name": "sdv21-dreambooth-lora-test",
        },
    )
    logger.debug(f"Submitted training job [id={thread_id}, response={response}, dirname={dirname}]")
    await thread.send(f"Submitted training job [id={thread.name}, model={response['job_id']}]")

    # Save the model
    MODEL_DB[thread_id] = LoRAPromptModel(
        thread_id=thread_id,
        thread_name=thread.name,
        model_id=f"custom/{response['job_id']}",
        prompt=prompt,
    )
    logger.debug(f"Saved model [id={thread_id}, model={MODEL_DB[thread_id]}]")

    if response is None:
        logger.error(f"Failed to submit training job [id={thread_id}, response={response}, dirname={dirname}]")
        await thread.send(f"Failed to train [prompt={prompt}, response={response}, dirname={dirname}]")


# Pull API token out of environment and run the bot:
bot_token = os.environ.get("DISCORD_BOT_TOKEN")
if bot_token is None:
    raise Exception("DISCORD_BOT_TOKEN environment variable not set")
logger.debug(f"Starting bot with token [token={bot_token[:5]}****]")
# bot.loop.run_until_complete(setup())
bot.run(bot_token)
