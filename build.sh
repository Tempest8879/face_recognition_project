#!/usr/bin/env bash
# ============================================================================
# Build script for Face Processor C++ module (Linux - uses g++)
# ============================================================================
set -e

echo "[1/3] Getting Python paths..."
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND11_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "   Python Include: $PYTHON_INCLUDE"
echo "   Pybind11 Include: $PYBIND11_INCLUDE"
echo "   Extension Suffix: $EXT_SUFFIX"

echo "[2/3] Compiling face_processor_cpp${EXT_SUFFIX} ..."
g++ -O2 -shared -fPIC -std=c++17 \
    -I"$PYTHON_INCLUDE" \
    -I"$PYBIND11_INCLUDE" \
    cpp/bindings.cpp \
    -o "python/face_processor_cpp${EXT_SUFFIX}"

echo ""
echo "============================================"
echo "  BUILD SUCCESSFUL!"
echo "  Output: python/face_processor_cpp${EXT_SUFFIX}"
echo "============================================"
