CC = gcc
CXX = g++
NVCC = nvcc

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler="-fopenmp"
LDFLAGS = -lm
EXES = nms-torch nms-baseline

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

nms-torch: nms-torch.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

nms-baseline: nms-baseline.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?
