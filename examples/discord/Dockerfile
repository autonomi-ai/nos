FROM python:3.8-slim

ENV PROJECT nos-bot

WORKDIR /tmp/$PROJECT
ADD requirements.txt .

# Install nos client and discord dependencies
RUN pip install -r requirements.txt

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "nos_bot.py"]