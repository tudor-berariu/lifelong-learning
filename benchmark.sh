#!/bin/bash

SCRIPT1=multi_task_classification.py -c benchmark --runs-no 72

liftoff $SCRIPT1 --procs-no 1 --comment "Single" || true
liftoff $SCRIPT1 --procs-no 2 --comment "2 in parallel" || true
liftoff $SCRIPT1 --procs-no 3 --comment "3 in parallel" || true
liftoff $SCRIPT1 --procs-no 4 --comment "4 in parallel" || true

SCRIPT2=multi_task_classification.py -c benchmark --runs-no 72

# Balance on GPUs
for pg in `seq 4`
do
    for omp in 1 2 4 8
    do
	liftoff $SCRIPT2 --gpus 0,1,2,3,4,5 --per-gpu $pg --omp $omp --mkl $mkl --procs-no 1000 --comment "pg=$pg;omp=$omp" || true
    done
done

