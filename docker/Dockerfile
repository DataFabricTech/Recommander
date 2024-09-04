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
FROM setter as tester

###############################################################
###                      Stage : Build                      ###
###############################################################
FROM setter as builder

ARG PROJECT_VERSION=v1.0.0

WORKDIR /app

RUN apt update && apt install -y build-essential \
	libpq-dev

RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

###############################################################
###             Stage : Make Product Image                  ###
###############################################################
# FROM ${IMAGE} as product
#
# ## Labelling
# LABEL company="Mobigen" \
#         team="mobigen-platform-team" \
#         email="irisdev@mobigen.com"
#
# WORKDIR /app
#
# ENV PROFILE "prod"
#
# RUN mkdir -p /app/conf
#
# ## Install Locale(ko_KR.UTF-8)
# RUN apt-get update && apt-get install -y locales \
#     && sed -i 's/^# \(ko_KR.UTF-8\)/\1/' /etc/locale.gen \
#     && locale-gen && localedef -f UTF-8 -i ko_KR ko_KR.UTF-8 \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*
#
# COPY --from=builder /app/build/libs/monitoring-1.0-SNAPSHOT.jar /app/service.jar
# COPY --from=builder /app/build/resources/main/* /app/conf/
#
# ENTRYPOINT ["java", "-Dspring.profiles.active=${PROFILE}", "-jar", "/app/service.jar"]
