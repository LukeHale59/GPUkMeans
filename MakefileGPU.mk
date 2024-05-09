CXX := nvc++#nvcc
CXXFLAGS := -fast #-O3 -std=c++11

all: kmeansGPU

kmeansGPU: kmeans-cpu.cu obj32
	$(CXX) $(CXXFLAGS) kmeans-gpu.cu -o obj32/kmeansGPU

obj32:  # Phony target to create the directory
	mkdir obj32

clean:
	rm -rf obj32 kmeansGPU
