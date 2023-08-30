ARG BASE_IMAGE
FROM ${BASE_IMAGE:-autonomi/nos:latest-cpu}

WORKDIR /tmp/$PROJECT
ADD requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY bot.py .
CMD ["python", "/app/bot.py"]