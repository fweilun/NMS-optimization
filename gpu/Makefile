CC = gcc
CXX = g++
NVCC = nvcc

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61  -Xcompiler="-fopenmp"
LDFLAGS = -lm
EXES = nms-torch

.PHONY: all clean

all: $(EXES)

clean:
	rm -f nms-torch nms-torch-test-2000 nms-torch-test nms-torch-solvebank nms-torch-solvebank64 nms-torch-solvebank-warp nms-padding

nms-torch: main.cpp nms-torch.cu nms.hpp
	$(NVCC) $(NVFLAGS) main.cpp nms-torch.cu -o $@ $(LDFLAGS)

nms-opt-2000: main.cpp nms-opt-2000.cu nms.hpp
	$(NVCC) $(NVFLAGS) main.cpp nms-opt-2000.cu -o $@ $(LDFLAGS)

nms-opt: main.cpp nms-opt.cu nms.hpp
	$(NVCC) $(NVFLAGS) main.cpp nms-opt.cu -o $@ $(LDFLAGS)

nms-torch-solvebank: main.cpp nms-torch-solvebank.cu nms.hpp
	$(NVCC) $(NVFLAGS) main.cpp nms-torch-solvebank.cu -o $@ $(LDFLAGS)

nms-torch-solvebank64: main.cpp nms-torch-solvebank64.cu nms.hpp
	$(NVCC) $(NVFLAGS) main.cpp nms-torch-solvebank64.cu -o $@ $(LDFLAGS)

nms-torch-solvebank-warp: main.cpp nms-torch-solvebank-warp.cu nms.hpp
	$(NVCC) $(NVFLAGS) main.cpp nms-torch-solvebank-warp.cu -o $@ $(LDFLAGS)

nms-padding: main.cpp nms-padding.cu nms.hpp
	$(NVCC) $(NVFLAGS) main.cpp nms-padding.cu -o $@ $(LDFLAGS)