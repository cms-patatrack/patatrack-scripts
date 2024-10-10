#! /bin/bash

EXEC="cmsRun"
FILE="/dev/stdout"
if [ "$1" ]; then
  FILE="$1"
fi

echo "timestamp,pid,%cpu,times,%mem,rss,vsz,args" > $FILE
while true; do
  TIME=$(printf "%.3f" $(date +%s.%N))
  ps h -C $EXEC -o 'pid,%cpu,times,%mem,rss,vsz,args' | sed -e"s/^ */$TIME,/;s/ \+/,/;s/ \+/,/;s/ \+/,/;s/ \+/,/;s/ \+/,/;s/ \+/,/" >> $FILE
  sleep 1s
done
