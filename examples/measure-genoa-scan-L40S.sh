#! /bin/bash

BASE="genoa-scan-L40S"
SLOT="n=4-7:m=4-7:nv=GPU-472e6c90-d880-44a5-7c4a-96722bdf74fa:amd="

start-mps-daemon.sh

eval `scram runtime -sh`

for J in {1..8}; do
  DOIO="--run-io-benchmark"
  for T in {1..10} {12..18..2} {20..32..4} {40..48..8}; do
    L=$(printf %02d $T)
    NAME=${BASE}-${J}x${L}-logs
    ./patatrack-scripts/benchmark reduced_hlt_10k.py \
      -e 10300 \
      --event-resolution 10 \
      --event-skip 300 \
      -r 3 \
      -j $J \
      -t $T \
      -s $T \
      -l ${NAME} \
      ${DOIO} \
      --csv ${NAME}.csv \
      --tmpdir '/fff/data' \
      --auto-delete 'run383631/*.dat' \
      --slot ${SLOT} |& tee ${NAME}.log
    DOIO="--no-run-io-benchmark"
  done
done

stop-mps-daemon.sh
