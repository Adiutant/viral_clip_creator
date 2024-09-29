FROM python:3.10-slim-buster

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt
RUN apt install ffmpeg

EXPOSE 5000

CMD ["python", "./web_inference.py"]