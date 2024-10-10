#! /bin/bash

function header() {
  printf "timestamp, "
  free -w | head -1 | sed -e's/^ *//' -e's/ \+/, /g'
}


function measure_one() {
  printf "%.3f, " $(date +%s.%N)
  free -w | sed -n -e'/^Mem:/s/ \+/, /g' -e's/^Mem:, //p'
}

function measure() {
  header
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
