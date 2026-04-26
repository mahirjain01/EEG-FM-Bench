#!/usr/bin/env bash
# Launch kNN model sweeps in parallel on specified GPUs.
set -e

REPO=/home/neurodx/mahir/EVAL_PAPER/EEG-FM-Bench
LOGDIR=$REPO/uniformevalbench/experiments/logs/knn
PYENV=/home/neurodx/arvasu/ndx-pipeline/venv/bin/python
SCRIPT=$REPO/uniformevalbench/experiments/axis_a_knn.py

source /home/neurodx/arvasu/ndx-pipeline/venv/bin/activate

export PYTHONPATH=$REPO:$PYTHONPATH
export EEGFM_PROJECT_ROOT=$REPO
export EEGFM_CONF_ROOT=$REPO/assets/conf
export EEGFM_DATABASE_PROC_ROOT=/mnt/eegfmbench/proc
export EEGFM_DATABASE_CACHE_ROOT=/mnt/eegfmbench/cache
export EEGFM_DATABASE_RAW_ROOT=/mnt/eegfmbench/raw
export EEGFM_RUN_ROOT=/mnt/eegfmbench/runs/uniformeval/axis_a_knn
# Prevent OpenBLAS from spinning up too many threads across parallel model processes
# (exceeded precompiled NUM_THREADS causes heap corruption during kNN)
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

MODEL=${1:-all}

launch_model() {
    local model=$1
    local gpu=$2
    local logfile=$LOGDIR/knn_${model}.log
    echo "Launching $model on GPU $gpu → $logfile"
    CUDA_VISIBLE_DEVICES=$gpu $PYENV $SCRIPT --model "$model" --in-process \
        >> "$logfile" 2>&1 &
    echo "$model PID=$!"
}

case "$MODEL" in
    csbrain) launch_model csbrain 3 ;;
    eegpt)   launch_model eegpt   4 ;;
    labram)  launch_model labram  5 ;;
    reve)    launch_model reve    7 ;;
    all)
        launch_model csbrain 3
        launch_model eegpt   4
        launch_model labram  5
        launch_model reve    7
        ;;
    *)
        echo "Usage: $0 [csbrain|eegpt|labram|reve|all]"
        exit 1
        ;;
esac

echo "All launched. Check $LOGDIR for logs."
