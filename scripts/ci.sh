#!/bin/sh

# Exit on error
set -e

echo "Running continuous integration..."

mkdir -p CI

python3 scripts/rand_data.py 1000 > CI/data.csv

cmake -S . -B CI/acpp -DCMAKE_BUILD_TYPE=RelWithDebInfo --log-level=ERROR
cmake --build CI/acpp --parallel 4
./CI/acpp/test-devices CI/data.csv 4

source /opt/intel/oneapi/setvars.sh || true
cmake -S . -B CI/oneapi -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKMEANS_USE_ONEAPI=ON -DCMAKE_CXX_COMPILER=icpx --log-level=ERROR
cmake --build CI/oneapi --parallel 4
./CI/oneapi/test-devices CI/data.csv 4

echo "Continuous integration passed successfully!"
