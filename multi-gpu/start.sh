#!/bin/sh
filename="main.py"
logger="logger.log"

if [ $# -eq 0 ]; then
    rm -f "$logger"
    touch "$logger"
else
    if [ $# -ge 1 ]; then
        filename=$1
    fi

    if [ $# -ge 2 ]; then
        logger=$2
    fi
fi

echo "deepspeed --num_gpus 2 "$filename" >> "$logger" 2>&1" >> "$logger" 2>&1
deepspeed --num_gpus 2 "$filename" >> "$logger" 2>&1