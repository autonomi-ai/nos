## Running the discord bot with skypilot

Follow the instructions in the discord bot [guide](../demos/discord-bot.md) to generate an API key, and make sure this is written to a `.env` file as `DISCORD_BOT_TOKEN=$YOUR_API_KEY`. Deploy the discord image generation bot via skypilot with the provided `server.yml`:

```bash
sky launch -c nos-server-gcp server.yaml --env-file=.env
```

Note that your chosen instance (GCP, AWS, Azure) must be able to reach the discord server where you want the bot deployed.
