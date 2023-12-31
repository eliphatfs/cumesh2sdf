mkdir -p bin
nvcc driver.cu -use_fast_math -std=c++17 --generate-line-info -o bin/mesh2sdf
