CC = gcc
CXX = g++
NVCC = nvcc

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler="-fopenmp"
LDFLAGS = -lm
EXES = nms-cpu nms-omp nms-SIMD

nms-baseline: main.cpp nms_baseline.cpp nms.hpp utils.cpp utils.hpp
	g++ -std=c++17 main.cpp nms_baseline.cpp utils.cpp -o nms-baseline.exe

nms-SoA: main.cpp nms_SoA.cpp nms.hpp utils.cpp utils.hpp
	g++ -std=c++17 -fopenmp main.cpp nms_SoA.cpp utils.cpp -o nms-SoA.exe

nms-omp: main.cpp nms_omp.cpp nms.hpp utils.cpp utils.hpp
	g++ -std=c++17 -fopenmp main.cpp nms_omp.cpp utils.cpp -o nms-omp.exe

nms-SIMD: main.cpp nms_SIMD.cpp nms.hpp utils.cpp utils.hpp
	g++ -std=c++17 -fopenmp -march=native main.cpp nms_SIMD.cpp utils.cpp -o nms-SIMD.exe

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)
