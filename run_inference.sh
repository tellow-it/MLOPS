#!/bin/bash
INFERENCE_DIR=$1
RESULT_FILE_PATH=$2
python -m src.inference -i $INFERENCE_DIR -o $RESULT_FILE_PATH