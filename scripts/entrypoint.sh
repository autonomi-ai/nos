#!/bin/sh

# Get number of cores
NCORES=$(nproc --all)
echo "Starting server with OMP_NUM_THREADS=${OMP_NUM_THREADS:-${NCORES}}..."
# Get OMP_NUM_THREADS from environment variable, if set otherwise use 1
OMP_NUM_THREADS=${OMP_NUM_THREADS:-${NCORES}} \
ray --logging-level warning \
    start --head \
    --port 6379 \
    --disable-usage-stats \
    --include-dashboard 0 \
    > /tmp/ray_server.log
nos-grpc-server
