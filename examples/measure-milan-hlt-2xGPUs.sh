#! /bin/bash

PDU=pdubk-c2b03-00-03

eval `scram runtime -sh`

# make sure the NVIDIA MPS service is running
start-mps-daemon.sh
echo

# keep track of the monitoring jobs to kill at the end of the measurement
PIDS=

function stop_monitoring_jobs() {
  # send the SIGINT signal
  kill -2 $PIDS >& /dev/null
  sleep 1
  PIDS=$(ps -o pid= $PIDS)

  # if the jobs are still running, send the SIGTERM signal
  if [ "$PIDS" ]; then
    kill    $PIDS >& /dev/null
    sleep 1
    PIDS=$(ps -o pid= $PIDS)
  fi

  # if the jobs are still running, send the SIGKILL signal
  if [ "$PIDS" ]; then
    kill    $PIDS >& /dev/null
    sleep 1
    PIDS=$(ps -o pid= $PIDS)
  fi

  # if the jobs are still running, print a message
  if [ "$PIDS" ]; then
    echo "Could not stop job(s) $PIDS"
  fi
}


function on_exit() {
  # stop the monitoring jobs
  stop_monitoring_jobs

  # stop the the NVIDIA MPS service
  stop-mps-daemon.sh
}

trap "on_exit" EXIT

# usage:
#  measure EVENTS SKIP NAME [LOGS=NAME]
#
# Benchmark NAME.py over EVENTS events, writing the measurements to LOGS/, LOGS.log, and LOGS-*.csv
function measure() {
  # expect integers
  EVENTS=$(( $1 ))
  SKIP=$(( $2 ))

  # expect the basename of a python file
  NAME="$3"
  [ -f "${NAME}.py" ] || { echo "Cannot find ${NAME}.py, aborting."; exit 1; }

  # if LOGS is not specified, use NAME
  LOGS="$4"
  [ "$LOGS" ] || LOGS="$NAME"

  # start the monitoring jobs
  PIDS=""
  mkdir -p ${LOGS}
  ./monitor_system.sh --pdu ${PDU}  ${LOGS}/system_monitor.csv >& /dev/null &   PIDS="$PIDS $!"
  ./monitor_cpu_power.sh            ${LOGS}/cpu_power.csv &                     PIDS="$PIDS $!"
  ./monitor_cpu_temp.sh             ${LOGS}/cpu_temp.csv &                      PIDS="$PIDS $!"
  ./monitor_cpu_memory.sh           ${LOGS}/cpu_memory.csv &                    PIDS="$PIDS $!"
  ./monitor_process_memory.sh       ${LOGS}/process_memory.csv &                PIDS="$PIDS $!"
  ./monitor_gpu.sh                  ${LOGS}/gpu_monitor.csv &                   PIDS="$PIDS $!"

  # let the monitoring jobs record the baseline before the measurement
  sleep 300s

  # perform the measurements
  ./patatrack-scripts/benchmark ${NAME}.py \
    -e $(($EVENTS + $SKIP)) --event-resolution 10 --event-skip $SKIP \
    -r 4 --wait 300 \
    -j 8 -t 32 -s 24 \
    -l ${LOGS} \
    --csv ${LOGS}/benchmark.csv \
    --tmpdir '/fff/data' \
    --auto-delete 'run383631/*.dat' \
    --slot n=0:m=0:nv=GPU-ba754dee-147f-b0e0-d00c-58b82ea964da \
    --slot n=1:m=1:nv=GPU-ba754dee-147f-b0e0-d00c-58b82ea964da \
    --slot n=2:m=2:nv=GPU-ba754dee-147f-b0e0-d00c-58b82ea964da \
    --slot n=3:m=3:nv=GPU-ba754dee-147f-b0e0-d00c-58b82ea964da \
    --slot n=4:m=4:nv=GPU-3b2bad55-4b37-d9f3-28c4-9e0d546ead4e \
    --slot n=5:m=5:nv=GPU-3b2bad55-4b37-d9f3-28c4-9e0d546ead4e \
    --slot n=6:m=6:nv=GPU-3b2bad55-4b37-d9f3-28c4-9e0d546ead4e \
    --slot n=7:m=7:nv=GPU-3b2bad55-4b37-d9f3-28c4-9e0d546ead4e \
    |& tee ${LOGS}/benchmark.log

  # let the monitoring jobs record the baseline after the measurement
  sleep 300s

  # stop the monitoring jobs
  stop_monitoring_jobs
}

# skip 20k events and measure over 50k events
measure 50000 20000 hlt milan-hlt-2xGPUs

# stop the the NVIDIA MPS service
stop-mps-daemon.sh
