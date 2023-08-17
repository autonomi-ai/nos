#!/usr/bin/env python

import os

import discord
from discord.ext import commands

import nos
from nos.client import InferenceClient, TaskType


# Init nos server
nos.init(runtime="gpu")
nos_client = InferenceClient()
nos_client.WaitForServer()
assert nos_client.IsHealthy()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="$", intents=intents)


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

    # Bit of a kluge but it seems ctx requires image to be a file that is opened through
    # the discrord API.
    tmp_file_path = "image.png"
    image.save(tmp_file_path)
    with open(tmp_file_path, "rb") as img_file:
        await ctx.send(file=discord.File(img_file))

    os.remove(tmp_file_path)


bot_token = os.environ.get("BOT_TOKEN")
if bot_token is None:
    raise Exception("BOT_TOKEN environment variable not set")

bot.run(bot_token)
