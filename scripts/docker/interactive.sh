#!/usr/bin/env bash

PORT=${PORT:-8889}

docker run --gpus=all -it --rm -e CUDA_VISIBLE_DEVICES --ipc=host -p $PORT:$PORT -v /home/:/home/ -v /efs/:/efs/ -v $PWD:/vocos/ vocos:latest bash 
