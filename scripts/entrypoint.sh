#!/bin/sh

# Get number of cores
NCORES=$(nproc --all)
echo "Starting server with OMP_NUM_THREADS=${OMP_NUM_THREADS:-${NCORES}}..."
# Get OMP_NUM_THREADS from environment variable, if set otherwise use 1
OMP_NUM_THREADS=${OMP_NUM_THREADS:-${NCORES}} ray start --head

echo "Starting NOS server..."
nos-grpc-server
