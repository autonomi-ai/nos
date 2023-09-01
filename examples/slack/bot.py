import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from flask import Flask, request, Response
import nos
from nos.client import InferenceClient, TaskType

app = Flask(__name__)

# Initialize the Slack WebClient
client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])

nos.init(runtime='gpu')

nos_client = InferenceClient()

BASE_MODEL = "runwayml/stable-diffusion-v1-5"

@app.route('/slack/events', methods=['POST'])
def command():
    print('we got an event!')
    data = request.json
    if 'challenge' in data:
        return Response(data['challenge'], mimetype='application/json')
    
    channel = data['event']['channel']
    text = data['event']['text']  
    user = data['event']['user']
    print(f'command: {command}')

    try:
        response = client.chat_postMessage(
            channel=channel,
            text=f"generate request: {text}"
        )
        nos_response = nos_client.Run(
            task=TaskType.IMAGE_GENERATION,
            model_name=BASE_MODEL,
            prompts=[text],
            width=512,
            height=512, 
            num_images=1,
        )
        (image,) = nos_response["images"]

        # save the image to disk
        image.save("generated.png")

        response = client.files_upload(
                channels=channel,
                file="generated.png",  # Replace with the actual path to your image
                title="Generated Image",
                initial_comment=f"Here's your generated image, <@{user}>!"
            )
        print(response)
    except SlackApiError as e:
        print(f"Error: {e.response['error']}")

    return Response(status=200)

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 3000)))