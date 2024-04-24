CXX := nvc++
CXXFLAGS := -fast

all: p2

p2: kmeans-gpu.cu
	$(CXX) $(CXXFLAGS) kmeans-gpu.cu -o obj32Serial/p2

clean:
	rm -rf obj32Serial p2
