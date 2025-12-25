CXX = g++
CXXFLAGS = -O3 -mcpu=native -march=native -mtune=native
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_80
LDFLAGS = -lcudart -lcublas

all: mygemm

main.o: main.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

mygemm: main.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f mygemm *.o