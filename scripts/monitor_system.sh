#! /bin/bash
PDU=

if [ "$1" == "--pdu" ]; then
  PDU="$2"
  shift 2
fi

if [ -z "$PDU" ]; then
  echo "Usage: $0 --pdu PDU [FILE]"
  exit 1
fi

if [ "$1" ]; then
  FILE="$1"
  exec ./pdu/rackWithPDU.sh -a wattscsv -o all -p $PDU -f $FILE -e -l 1
else
  exec ./pdu/rackWithPDU.sh -a watts -o all -p $PDU -e -l 1
fi
