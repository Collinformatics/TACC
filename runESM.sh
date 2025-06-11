#!/bin/bash

inModelType='3B Params'
inEnzymeName='Mpro2'
inUseReadingFrame=true
inMinSubs=100

# Get inputs
while getopts "r:p:l" opt; do
  case $opt in
    r)
      AA=$OPTARG
      ;;
    p)
      pos=$OPTARG
      ;;
    l)
      inModelType='15B Params'
      ;;
    *)
      exit 1
      ;;
  esac
done

python ESM.py "$inModelType" "$inEnzymeName" "$AA" "$pos" "$inUseReadingFrame" "$inMinSubs"
