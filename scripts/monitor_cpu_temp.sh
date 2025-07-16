#! /bin/bash

function header() {
  printf "timestamp"
  sensors -j 2> /dev/null | jq '.[] | to_entries[] | select(.key | startswith("Tccd")) | .key' | xargs printf ', %s'
  echo
}

function measure_one() {
  printf "%.3f" $(date +%s.%N)
  sensors -j 2> /dev/null | jq '.[] | to_entries[] | select(.key | startswith("Tccd")) | .value | to_entries[] | select(.key | startswith("temp")) | .value' | xargs printf ', %.1f'
  echo
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
