CXX := nvc++#nvcc
CXXFLAGS := -fast #-O3 -std=c++11

all: p2

p2: kmeans-gpu.cu obj32Serial
	$(CXX) $(CXXFLAGS) kmeans-gpu.cu -o obj32Serial/p2

obj32Serial:  # Phony target to create the directory
	mkdir obj32Serial

clean:
	rm -rf obj32Serial p2
