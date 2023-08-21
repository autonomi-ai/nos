#!/usr/bin/env python

import io
import os

import discord
from discord.ext import commands

import nos
from nos.client import InferenceClient, TaskType


# Init nos server, wait for it to spin up then confirm its healthy:
nos.init(runtime="gpu")
nos_client = InferenceClient()
nos_client.WaitForServer()
if not nos_client.IsHealthy():
    raise RuntimeError("NOS server is not healthy")

# Set permissions for our bot to allow it to read messages:
intents = discord.Intents.default()
intents.message_content = True

# Create our bot:
bot = commands.Bot(command_prefix="$", intents=intents)

# Create a callback to read messages and generate images from prompt:
@bot.command()
async def generate(ctx, *, prompt):
    response = nos_client.Run(
        TaskType.IMAGE_GENERATION,
        "stabilityai/stable-diffusion-2",
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


# Pull API token out of environment and run the bot:
bot_token = os.environ.get("BOT_TOKEN")
if bot_token is None:
    raise Exception("BOT_TOKEN environment variable not set")

bot.run(bot_token)
