#!/bin/sh

# Get number of cores
NCORES=$(nproc --all)

# Get OMP_NUM_THREADS from environment variable, if set otherwise use 1
echo "Starting server with OMP_NUM_THREADS=${OMP_NUM_THREADS:-${NCORES}}..."
OMP_NUM_THREADS=${OMP_NUM_THREADS:-${NCORES}} \
ray --logging-level warning \
    start --head \
    --port 6379 \
    --disable-usage-stats \
    --include-dashboard 0 \
    > /tmp/ray_server.log

# Start the server (as a background process) and write the server's logs to a file
# --address ${NOS_GRPC_HOST:-"[::]"}:${NOS_GRPC_PORT:-50051} \
echo "Starting gRPC server..."
nos-grpc-server \
    2>&1 | tee -a /tmp/grpc_server.log &

# Start the client if --http flag is provided to this script
if [ "$1" = "--http" ]; then
    echo "Starting HTTP proxy server"
    nos-http-server \
        --host ${NOS_HTTP_HOST:-0.0.0.0} \
        --port ${NOS_HTTP_PORT:-8000} \
        --workers ${NOS_HTTP_WORKERS:-1} \
        2>&1 | tee -a /tmp/http_server.log &
fi

# Wait for both background processes to finish
wait
