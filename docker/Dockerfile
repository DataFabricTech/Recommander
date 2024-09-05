ARG IMAGE=python:3.12-slim

##############################################################
###             Stage : Source Copy              ###
##############################################################
FROM ${IMAGE} as setter

LABEL name="Recommender Image"

LABEL service.name="Recommender"

WORKDIR /app

COPY ../src/app ./src/app
COPY ../src/common ./src/common
COPY ../src/main.py ./src/main.py
COPY ../config_templates/config.yml ./config_templates/config.yml
COPY ../requirements.txt ./

###############################################################
###             Stage : Source Analysis & Test              ###
###############################################################
# FROM setter as tester

###############################################################
###                      Stage : Build                      ###
###############################################################
# FROM setter as builder

###############################################################
###             Stage : Make Product Image                  ###
###############################################################
FROM setter as product

ARG PROJECT_VERSION=v1.0.0

LABEL company="Mobigen" \
        team="mobigen-platform-team" \
        email="irisdev@mobigen.com"

ENV PROFILE "prod"

WORKDIR /app

RUN apt update && apt install -y build-essential \
	libpq-dev

RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "src/main.py"]
