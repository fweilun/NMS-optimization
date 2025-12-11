CC = gcc
CXX = g++
NVCC = nvcc

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler="-fopenmp"
LDFLAGS = -lm
EXES = nms-torch

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

nms-torch: main.cpp nms-torch.cu nms.hpp
	$(NVCC) $(NVFLAGS) main.cpp nms-torch.cu -o $@ $(LDFLAGS)

nms-torch-test-2000: main.cpp nms-torch-test-2000.cu nms.hpp
	$(NVCC) $(NVFLAGS) main.cpp nms-torch-test-2000.cu -o $@ $(LDFLAGS)

nms-torch-test: main.cpp nms-torch-test.cu nms.hpp
	$(NVCC) $(NVFLAGS) main.cpp nms-torch-test.cu -o $@ $(LDFLAGS)

nms-padding: main.cpp nms-padding.cu nms.hpp
	$(NVCC) $(NVFLAGS) main.cpp nms-padding.cu -o $@ $(LDFLAGS)