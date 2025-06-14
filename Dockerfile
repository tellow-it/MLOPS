ARG BASE_IMAGE=python:3.12-slim
FROM $BASE_IMAGE

RUN apt-get update && apt-get install -y git

# system update & package install
RUN apt-get -y update && \
    apt-get install libpq-dev -y && \
    apt-get install libgl1-mesa-glx -y

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .

# pip & requirements
RUN python3 -m pip install --user --upgrade pip
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install -r requirements.txt