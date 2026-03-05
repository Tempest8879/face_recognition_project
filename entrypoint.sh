#!/usr/bin/env bash
set -e

# Build the C++ module if it hasn't been compiled yet
SO_FILE=$(python3 -c "import sysconfig; print('python/face_processor_cpp' + sysconfig.get_config_var('EXT_SUFFIX'))")
if [ ! -f "$SO_FILE" ]; then
    echo "[entrypoint] Building C++ module..."
    bash build.sh
fi

echo "[entrypoint] Starting Face Recognition API on port ${PORT:-8000}..."
exec uvicorn python.api_server:app --host 0.0.0.0 --port "${PORT:-8000}"
