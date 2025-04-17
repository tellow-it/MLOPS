#!/bin/bash
EXPERIMENT_INFO=$1
dvc repro || exit 1
git add .
git commit -m "Experiment: $EXPERIMENT_INFO"
COMMIT=`git rev-parse HEAD`
python -m src.track_commit $COMMIT