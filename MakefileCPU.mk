CXX := nvc++#nvcc
CXXFLAGS := -fast #-O3 -std=c++11

all: kmeansCPU

kmeansCPU: kmeans-cpu.cu obj32
	$(CXX) $(CXXFLAGS) kmeans-cpu.cu -o obj32/kmeansCPU

obj32:  # Phony target to create the directory
	mkdir obj32

clean:
	rm -rf obj32 kmeansCPU
