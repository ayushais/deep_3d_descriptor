#!/bin/bash
cd thrift-0.11.0
NUM_MAKE_THREADS=$(nproc)
PROJECT_DIR="$(pwd)" 
./bootstrap.sh
./configure --prefix=$PROJECT_DIR --without-java PY_PREFIX="$PROJECT_DIR"
make -j${NUM_MAKE_THREADS}
make install

