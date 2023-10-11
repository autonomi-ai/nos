# Building a MidJourney clone with NOS + Discord

1. **Registering a discord bot to create our API key**

    This can be done through the discord developer [guide](https://discord.com/developers/docs/getting-started). You will need
    a discord account as well as a server you wish to add the bot to. When we're finished, `nos-bot` will accept image generation requests
    from users on this server and post the resulting images to the main channel. Once you have your bot API key, add it to your local
    environment like so:
    ```bash
    export BOT_TOKEN=$YOUR_DISCORD_API_TOKEN
    ```

2. **Setting up a NOS client to generate images**

    NOS comes with an endpoint for Stable Diffusion V2 from HuggingFace, so all we need to do is init a server on our machine and verify we can
    connect to it from the client:
    ```python
    import nos
    from nos.client import Client, TaskType
    import os

    # Init nos server
    nos.init(runtime='gpu')
    nos_client = Client()
    nos_client.WaitForServer()
    assert nos_client.IsHealthy()
    ```

    See `examples/notebook/inference-client-example.ipynb` for a better overview. Nos will initialize a GPU-ready container on our machine and return
    to the client that it's ready to go.

2. **The Discord interface**

    Next we need a way to handle message requests on our server. Discord.py makes callbacks pretty easy:

    ```python
    intents = discord.Intents.default()
    intents.message_content = True

    bot = commands.Bot(command_prefix='$', intents=intents)

    @bot.command()
    async def generate(ctx, *, prompt):
        response = nos_client.Run(TaskType.IMAGE_GENERATION, "stabilityai/stable-diffusion-2",
                            prompts=[prompt], width=512, height=512, num_images=1)
        image, = response["images"]

        tmp_file_path = "image.png"
        image.save(tmp_file_path)
        with open(tmp_file_path, "rb") as img_file:
            await ctx.send(file=discord.File(img_file))

        os.remove(tmp_file_path)
    ```

    We need the `message_content` intent so we can access the contents of user messages to retrieve image prompts. We'll parse generation requests
    as `$generate image prompt here...`. Any messages beginning with `$generate` will be sent to Nos for image generation. The rest of our
    message handler is pretty straightforward: we run the client (with `TaskType.IMAGE_GENERATION`) to produce a set of images, then we retrieve
    the first result from the list, save it locally, and call `ctx.send` to upload the image with the discord `File` interface.

3. **Time to run the server**

    The full server is only ~40 LOC:
    ```python
    #!/usr/bin/env python

    import discord
    from discord.ext import commands

    import nos
    from nos.client import Client, TaskType
    import os

    # Init nos server
    nos.init(runtime='gpu')
    nos_client = Client()
    nos_client.WaitForServer()
    assert nos_client.IsHealthy()

    intents = discord.Intents.default()
    intents.message_content = True

    bot = commands.Bot(command_prefix='$', intents=intents)

    @bot.command()
    async def generate(ctx, *, prompt):
        response = nos_client.Run(TaskType.IMAGE_GENERATION, "stabilityai/stable-diffusion-2",
                                prompts=[prompt], width=512, height=512, num_images=1)
        image, = response["images"]

        tmp_file_path = "image.png"
        image.save(tmp_file_path)
        with open(tmp_file_path, "rb") as img_file:
            await ctx.send(file=discord.File(img_file))

        os.remove(tmp_file_path)

    bot_token = os.environ.get("BOT_TOKEN")
    if bot_token is None:
        raise Exception("BOT_TOKEN environment variable not set")

    bot.run(bot_token)
    ```


    We should be call set. You can try out the whole thing with `python nos/experimental/nos_bot.py`


    ![Bot Running](./discord-bot/discord-bot-demo.png)
