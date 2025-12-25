CXX = g++
CXXFLAGS = -O3 -mcpu=native -march=native -mtune=native
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_80
LDFLAGS = -lcudart -lcublas

all: mygemm

KERNEL_FILES = kernel1.cpp kernel2.cpp kernel3.cpp kernel4.cpp kernel5.cpp kernel6.cpp
KERNEL_OBJS = $(KERNEL_FILES:.cpp=.o)

%.o: %.cpp
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS)

gemm.o: gemm.cpp
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS)

main.o: main.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

mygemm: main.o gemm.o $(KERNEL_OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f mygemm *.o