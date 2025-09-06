#!/bin/bash

#Please login first if haven't done so
#   huggingface-cli login
#   input your token
# To Generate a token: https://huggingface.co/settings/tokens
# Then configure git:
#git config --global credential.helper store
#echo "https://<YOUR_TOKEN>@huggingface.co" > ~/.git-credentials
# ---------------
hf auth login --token $HF_TOKEN --add-to-git-credential
set -euo pipefail
# Default values
APPEND_DATE=false
DATASET_NAME=""
OUTPUT="dataset"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET_NAME="$2"
      shift 2
      ;;
    --append-date)
      APPEND_DATE=true
      shift
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
  esac
done

if [[ -z "$DATASET_NAME" ]]; then
  echo "Error: --dataset is required"
  exit 1
fi

# Construct output directory and log filename
if $APPEND_DATE; then
  DATESTAMP="$(date '+%Y-%m-%d')"
  OUTPUT="${OUTPUT}-${DATESTAMP}"
else
  DATESTAMP="$(date '+%Y-%m-%d')"
fi

LOGFILE="${OUTPUT}.log"
echo "Downloading $DATASET_NAME"
echo "Logs will go to: $LOGFILE"

# Run the actual job under nohup and detach
function download_dataset() {
    echo "Started downloading `date`"
    git clone https://huggingface.co/datasets/$1 $2
    echo "Finished git clone `date`"
    cd $2
    git lfs pull
    echo 'Finished pull `date`'
}
export  -f download_dataset
nohup bash -c "download_dataset $DATASET_NAME $OUTPUT" >& $LOGFILE & 
