#!/usr/bin/env python

import io
import os

import discord
from discord.ext import commands

import nos
from nos.client import InferenceClient, TaskType
from nos.constants import NOS_TMP_DIR


# Init nos server, wait for it to spin up then confirm its healthy:
nos_client = InferenceClient()
nos_client.WaitForServer()
if not nos_client.IsHealthy():
    raise RuntimeError("NOS server is not healthy")

# Set permissions for our bot to allow it to read messages:
intents = discord.Intents.default()
intents.message_content = True

# Create our bot:
bot = commands.Bot(command_prefix="$", intents=intents)

TRAINING_CHANNEL_NAME = "training"
NOS_TRAINING_DIR = NOS_TMP_DIR / "train"

# Init a dictionary to map job ids to thread ids:
thread_to_job = {}

# Create a callback to read messages and generate images from prompt:
@bot.command()
async def generate(ctx, *, prompt):
    # Pull the thread id so we know which model to run generation on:
    if ctx.channel.id not in thread_to_job:
        await ctx.send("No thread found for this channel, please train a model first!")
        return

    job_id = thread_to_job.get(ctx.channel.id)

    # TODO (Sudeep/Scott): What is the setup for training given the job id? 
    model_from_job_id = "custom/" + job_id
    response = nos_client.Run(
        TaskType.IMAGE_GENERATION,
        model_from_job_id,
        prompts=[prompt],
        width=512,
        height=512,
        num_images=1,
    )
    image = response["images"][0]

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    await ctx.send(file=discord.File(image_bytes, filename="image.png"))

 
@bot.command()
async def train(ctx):
    # check that its in the training channel
    if ctx.channel.name != TRAINING_CHANNEL_NAME:
        print("not in training channel, returning!")
        return

    if not ctx.message.attachments:
        print("no attachments to train on, returning!")
        return

    # create a thread for this training job:
    thread_name = str(ctx.message.id)
    thread = await ctx.channel.create_thread(name=thread_name, type=discord.ChannelType.public_thread)

    await thread.send(f"Created a new thread: {thread.name}")

    dirname = NOS_TRAINING_DIR / thread_name
    dirname.mkdir(parents=True, exist_ok=True)

    await thread.send("saving at dir: " + str(dirname))

    # save the attachments
    for attachment in ctx.message.attachments:
        print(f"got attachement: {attachment.filename}")
        await attachment.save(os.path.join(dirname, attachment.filename))
        await thread.send(f"Image {attachment.filename} saved!")

    # Kick off a nos training run
    from nos.server._service import TrainingService

    svc = TrainingService()
    job_id = svc.train(
        method="stable-diffusion-dreambooth-lora",
        training_inputs={
            "model_name": "stabilityai/stable-diffusion-2-1",
            "instance_directory": dirname,
        },
        metadata={
            "name": "sdv21-dreambooth-lora-test-bench",
        },
    )
    assert job_id is not None

    thread.send(f"Started training job: {job_id}")
    job_to_thread[job_id] = thread


# Pull API token out of environment and run the bot:
bot_token = os.environ.get("BOT_TOKEN")
if bot_token is None:
    raise Exception("BOT_TOKEN environment variable not set")

bot.run(bot_token)
