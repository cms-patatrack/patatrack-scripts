#! /bin/bash

if [ "$1" ]; then
  FILE="$1"
  nvidia-smi --query-gpu=timestamp,uuid,name,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv --loop=1 --filename=$FILE
else
  nvidia-smi --query-gpu=timestamp,uuid,name,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv --loop=1
fi
