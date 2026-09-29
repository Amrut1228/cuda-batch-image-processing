#!/bin/bash

set -e

echo "Building CUDA batch signal processor..."
make clean
make

echo
echo "Running CUDA batch signal processor..."
./signal_batch --signals 512 --samples 4096 --radius 2
