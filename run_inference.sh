#!/bin/bash
INFERENCE_FILE_PATH=$1
RESULT_FILE_PATH=$2
dvc repro || exit 1
git add .
git commit -m "Experiment: $EXPERIMENT_INFO"
COMMIT=`git rev-parse HEAD`
python -m src.track_commit $COMMIT