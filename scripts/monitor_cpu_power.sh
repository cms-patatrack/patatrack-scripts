#! /bin/bash

function measure_one() {
  printf "%.3f" $(date +%s.%N)
  for P in /sys/class/powercap/intel-rapl/intel-rapl:*; do
    NAME=$(< $P/name)
    ENERGY=$(echo $(< $P/energy_uj) / 1000000 | bc -l)
    printf ", %s, %.2f" $NAME $ENERGY
  done
  echo
}

function measure() {
  while true; do
    measure_one
    sleep 1s
  done
}

if [ "$1" ]; then
  FILE="$1"
  measure > $FILE
else
  measure
fi
