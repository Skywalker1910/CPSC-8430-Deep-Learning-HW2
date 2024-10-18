#!/bin/bash
# Script to run seq2seq model
# Usage: ./hw2_seq2seq.sh <data_directory> <output_file>

data_dir=$1
output_file=$2

# Run the seq2seq model inference
python3 run_inference.py --data_dir $data_dir --output $output_file
