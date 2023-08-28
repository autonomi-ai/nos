#!/bin/sh
set -e
set -x

echo "Starting Ray server with OMP_NUM_THREADS=${OMP_NUM_THREADS}..."
# Get OMP_NUM_THREADS from environment variable, if set otherwise use 1
OMP_NUM_THREADS=${OMP_NUM_THREADS} ray start --head

echo "Starting NOS server..."
nos-grpc-server && python ./nos_bot.py
