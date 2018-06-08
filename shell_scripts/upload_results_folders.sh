#!/bin/bash

# Copy all folders with pattern '[0-9]*_*' from RES_FOLDER and show which folders were not copied
# Execute with: ./upload_results_folders.sh

RES_FOLDER=results/
TIMESTAMP=$(date +%s)
OUTFILE="$RES_FOLDER/$(date +%s)_server_upload.out"
OUTFILE2=$OUTFILE

find $RES_FOLDER -maxdepth 1 -type d -name '[0-9]*_*' -exec rsync --ignore-times \
    --checksum --progress  \
    -avr {}  tempuser@141.85.232.73:/home/tempuser/workspace/andrei/lifelong_learning_results \; \
    | tee $OUTFILE2;

echo ""
echo "Difference in local results and successfully copied folders:"

diff <(awk '/sending incremental file list/{getline; print}' $OUTFILE2 | sed 's#/##') \
    <(find $RES_FOLDER -type d -name '[0-9]*_*' | cut -sd / -f 2-);
