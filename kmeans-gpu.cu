// Implementation of the KMeans Algorithm
// reference: https://github.com/marcoscastro/kmeans

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <xmmintrin.h>
#include <limits>
#include "config_t.h"
#include <cassert>
#include <cstdlib>
#include <string>
#include <unistd.h>


using namespace std;

void parseargs(int argc, char** argv, config_t& cfg) {
    // parse the command-line options
    int opt;
    while ((opt = getopt(argc, argv, "k:s:t:p:h")) != -1) {
        switch (opt) {
          case 'k': cfg.clusters = atoi(optarg); break;
          case 's': cfg.seed = atoi(optarg); break;
		  case 't': cfg.threads = atoi(optarg); break;
		  case 'p': cfg.total_points = atoi(optarg); break;
		  case 'h': cfg.seed2 = atoi(optarg); break;
        }
    }
}

struct PointStruct {
    int id_cluster;
    float* values;

	PointStruct() : id_cluster(-1), values(nullptr) {}

    // Constructor to initialize values array with a specified size
    PointStruct(int size) : id_cluster(-1) {
        values = new float[size];
    }

    // Destructor to free memory allocated for values array
    ~PointStruct() {
        delete[] values;
    }
};

struct ClusterStruct {
    int numPoints;
    float* central_values;
    float* central_values_sums;

	ClusterStruct() : numPoints(0), central_values(nullptr), central_values_sums(nullptr) {}
    // Constructor to initialize central_values and central_values_sums arrays with a specified size
    ClusterStruct(int size) : numPoints(0){
        numPoints = 0;
        central_values = new float[size];
        central_values_sums = new float[size];
    }

    // Destructor to free memory allocated for central_values and central_values_sums arrays
    ~ClusterStruct() {
        delete[] central_values;
        delete[] central_values_sums;
    }

};

__global__ void distence_kernel_first(float* points,float* clusters, int total_points, int total_values, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    //increment by stride
	for(int i = idx; i < total_points; i+= stride)
	{
			float sum = 0.0, min_dist;
			int id_cluster_center = 0;
			for(int j = 0; j < total_values; j++)
			{
				sum += pow(clusters[1+j] -
						   points[i*(total_values+1) +j + 1], 2.0);
			}
			min_dist = sum;
			for(int m = 1; m < K; m++)
			{
				float dist;
				sum = 0.0;
				for(int j = 0; j < total_values; j++)
				{
					dist = clusters[m*(total_values*2 +1) +j + 1] -points[i*(total_values+1) +j + 1];
					sum += dist * dist;
				}
	    	    //remove the sqrt
				if(sum < min_dist)
				{
					min_dist = sum;
					id_cluster_center = m;
				}
			}
			points[i*(total_values+1)] = id_cluster_center;
	}
}

__global__ void clear_cluster_kernel(float* clusters, int total_values, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    //increment by stride
			//clear each cluster value store
	for(int i = idx; i < K; i += stride){
		for(size_t j = 0 ; j < total_values;j++){
			clusters[i*(total_values*2 +1) +j + 1+ total_values]=0;
		}
        clusters[i * (total_values*2 +1)] = 0;
    }
}

//serial version
// __global__ void sum_center_kernel(float* points,float* clusters, int total_points, int total_values)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     //int stride = gridDim.x * blockDim.x;
// 	//this needs to be fixed
// 	if(idx == 0){
//     	for(int i = 0; i < total_points; i++){
// 			int clusterID = (int)points[i*(total_values+1)];
//     	    for(int j = 0; j < total_values; j++){
// 				clusters[clusterID*(total_values*2 +1) +j + 1+ total_values] += points[i*(total_values+1) +j + 1];
//     	   	}
//     	clusters[clusterID * (total_values*2 +1)]++;
// 		}
// 	}
// }

//atomic version
__global__ void sum_center_kernel(float* points,float* clusters, int total_points, int total_values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(int i = idx; i < total_points; i+= stride){
		int clusterID = (int)points[i*(total_values+1)];
        for(int j = 0; j < total_values; j++){
			atomicAdd(&clusters[clusterID * (total_values * 2 + 1) + j + 1 + total_values], points[i * (total_values + 1) + j + 1]);
       	}
    atomicAdd(&clusters[clusterID * (total_values*2 +1)],1);
	}
}

//reduce version
// __global__ void sum_center_kernel(float* points,float* clusters, int total_points, int total_values, int K)
// {
// 	extern __shared__ float clustersLocal[];
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = gridDim.x * blockDim.x;
// 	unsigned int tid = threadIdx.x;
// 	//this needs to be fixed
//     for(int i = idx; i < total_points; i+= stride){
// 		int clusterID = (int)points[i*(total_values+1)];
//         for(int j = 0; j < total_values; j++){
// 			clustersLocal[clusterID*(total_values +1) +j + 1 +((K * total_values+ K)*stride )] += points[i*(total_values+1) +j + 1];
//        	}
//     	clustersLocal[clusterID*(total_values +1) +((K * total_values+ K)*stride )]++;
// 	}
// 	__syncthreads();
// 	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
// 		if (tid < s) {
// 			for(int i = 0 ; i < K;i++){
// 				for(int j = 0; j < total_values; j++){
// 					clustersLocal[i*(total_values +1) +j + 1 +((K * total_values+ K)*tid )] += clustersLocal[i*(total_values +1) +j + 1 +((K * total_values+ K)*(tid+2) )];
//        			}
// 				clustersLocal[i*(total_values +1) +((K * total_values+ K)*tid )] += clustersLocal[i*(total_values +1) +((K * total_values+ K)*(tid+2) )];
// 			}
// 		}
// 		__syncthreads();
// 	}
// 	if(tid == 0){
// 		for(int i = 0 ; i < K;i++){
// 			for(int j = 0; j < total_values; j++){
// 				clusters[i*(total_values*2 +1) +j + 1+ total_values] = clustersLocal[i*(total_values +1) +j + 1];
//     		}
// 		}
// 	}
// }

// template <unsigned int blockSize>
// __device__ void warpReduce(volatile int *sdata, unsigned int tid) {
// 	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
// 	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
// 	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
// 	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
// 	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
// 	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
// }


// template <unsigned int blockSize>
// __global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
// 	extern __shared__ int sdata[];
// 	unsigned int tid = threadIdx.x;
// 	unsigned int i = blockIdx.x*(blockSize*2) + tid;
// 	unsigned int gridSize = blockSize*2*gridDim.x;
// 	sdata[tid] = 0;
// 	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
// 	__syncthreads();
// 	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
// 	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
// 	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
// 	if (tid < 32) warpReduce(sdata, tid);
// 	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
// }


__global__ void get_center_kernel(float* clusters, int total_values, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    //increment by stride
			//clear each cluster value store
	for(int i = idx; i < K; i += stride){
		int total_points_cluster = clusters[i * (total_values*2 +1)];
        for(int j = 0; j < total_values; j++){
            float sum = clusters[i*(total_values*2 +1) +j + 1+ total_values];
            clusters[i*(total_values*2 +1) +j + 1] = sum / total_points_cluster;
        }
    }
}

__global__ void distence_kernel_main(float* points,float* clusters, int total_points, int total_values, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    //increment by stride
	for(int i = idx; i < total_points; i+= stride)
	{
		int id_old_cluster = points[i*(total_values+1)];
		float sum = 0.0, min_dist;
		int id_cluster_center = 0;
		for(int j = 0; j < total_values; j++)
		{
			sum += pow(clusters[1+j] -
				   points[i*(total_values+1) +j + 1], 2.0);
		}
		min_dist = sum;
		for(int m = 1; m < K; m++)
		{
			float dist;
			sum = 0.0;
			for(int j = 0; j < total_values; j++)
			{
				dist = clusters[m*(total_values*2 +1) +j + 1] -points[i*(total_values+1) +j + 1];
				sum += dist * dist;
			}
    	    //remove the sqrt
			if(sum < min_dist)
			{
				min_dist = sum;
				id_cluster_center = m;
			}
		}
		if(id_old_cluster != id_cluster_center)
		{
			points[i*(total_values+1)] = id_cluster_center;
			//done = false;
		}
	}
}

int main(int argc, char *argv[])
{
	// Parse command line arguments using getopt()
	config_t config;
    parseargs(argc, argv, config);
	//srand (time(NULL));
    srand (config.seed);

	int total_points, total_values, K = config.clusters, max_iterations, has_name;
	cin >> total_points >> total_values >> max_iterations >> has_name;

	if(config.total_points != -1){
		total_points = config.total_points;
	}

	//Once we know how big GPU memory is we can
	//cuda malloc as these calls are async and can be done while we are reading the data from file
	float* device_points;
    cudaMalloc(&device_points, sizeof(float) * total_points * total_values + sizeof(float) * total_points);

    float* device_clusters;
    cudaMalloc(&device_clusters, sizeof(float) * K * total_values *2 + sizeof(float) * K);
    
	//PointStruct* points = (PointStruct*)malloc(total_points * sizeof(PointStruct));

	float* points = (float*)malloc(sizeof(float) * total_points * total_values + sizeof(float) * total_points);

	// Initialize each PointStruct object
	for(int i = 0; i < total_points; i++) {
	    // Initialize the PointStruct object with the size of values array
	    // Fill in the values for each 'values' array
	    for(int j = 0; j < total_values; j++) {
	        float value;
	        cin >> value;
	        points[i*(total_values+1) +j + 1] = value;
	    }
	}

	//ClusterStruct* clusters = (ClusterStruct*)malloc(K * sizeof(ClusterStruct(total_values)));
	//values first, sums seconds
	float* clusters = (float*)malloc(sizeof(float) * K * total_values *2 + sizeof(float) * K);

	// std::chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now();
    
	if(K > total_points)
		return -1;
	vector<int> prohibited_indexes;

	// choose K distinct values for the centers of the clusters
	for(int i = 0; i < K; i++)
	{
		//new (&clusters[i]) ClusterStruct(total_values);
		while(true)
		{
			int index_point = rand() % total_points;
			if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
					index_point) == prohibited_indexes.end())
			{
				prohibited_indexes.push_back(index_point);
				points[index_point*(total_values+1)]=i;
				clusters[i * (total_values*2 +1)] = 0;
				for(size_t j = 0;j<total_values;j++){
					clusters[i*(total_values*2 +1) +j + 1] = points[index_point*(total_values+1) +j + 1];
					clusters[i*(total_values*2 +1) +j + 1+ total_values]=0;
				}
				break;
			}
		}
	}
	//we now need to GPU

	std::chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now();
    int threads_per_block = 512;
    int deviceId;
    cudaGetDevice(&deviceId);
  
    int numberOfSMs;
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    
    int number_of_blocks = 32 * numberOfSMs;
	cudaMemcpy(device_points, points, sizeof(float) * total_points * total_values + sizeof(float) * total_points, cudaMemcpyHostToDevice);
	cudaMemcpy(device_clusters, clusters, sizeof(float) * K * total_values *2 + sizeof(float) * K, cudaMemcpyHostToDevice);

	std::chrono::high_resolution_clock::time_point afterCudaCopyToDevice = chrono::high_resolution_clock::now();

	//assign each point to nearest cluster
	distence_kernel_first<<<number_of_blocks, threads_per_block>>>(device_points,device_clusters,total_points,total_values,  K);

	int iter = 2;
	while(true)
	{
		//clear each cluster value store
		clear_cluster_kernel<<<number_of_blocks, threads_per_block>>>(device_clusters,total_values,  K);

		//get the sum of all points in the cluster
		sum_center_kernel<<<number_of_blocks, threads_per_block>>>(device_points,device_clusters,total_points,total_values);

		//get the new centers for each cluster
		get_center_kernel<<<number_of_blocks, threads_per_block>>>(device_clusters,total_values,  K);

		//assign each point to nearest cluster
		distence_kernel_main<<<number_of_blocks, threads_per_block>>>(device_points,device_clusters,total_points,total_values,  K);
		
		if( iter == 10)
		{
			//cout << "Break in iteration " << iter << "\n\n";
			break;
		}
		iter++;
	}
	cudaDeviceSynchronize();
	std::chrono::high_resolution_clock::time_point afterKernels = chrono::high_resolution_clock::now();

	cudaMemcpy(clusters, device_clusters, sizeof(float) * K * total_values *2 + sizeof(float) * K, cudaMemcpyDeviceToHost);
	cudaMemcpy(points, device_points, sizeof(float) * total_points * total_values + sizeof(float) * total_points, cudaMemcpyDeviceToHost);

    std::chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();

	// shows elements of clusters
	for(int i = 0; i < K; i++)
	{
		int total_points_cluster =  clusters[i * (total_values*2 +1)];
		cout << "Cluster " << i + 1 << endl;
		cout << "total_points_cluster " << total_points_cluster << endl;
		cout << "Cluster values: ";
		for(int j = 0; j < total_values; j++)
			cout << clusters[i*(total_values*2 +1) +j + 1] << " ";
		cout << endl;
	}
	cout << "Time for Cuda Mem Copy Host To Device :" <<std::chrono::duration_cast<std::chrono::milliseconds>(afterCudaCopyToDevice-begin).count() <<" ms"<<endl;
	//this is in micro seconds because it is so fast
	float us = (float) std::chrono::duration_cast<std::chrono::microseconds>(afterKernels-afterCudaCopyToDevice).count();
	float ms = us/1000;
	cout << "Time for Kernels (k-means) :" <<ms <<" ms"<<endl;
	cout << "Time for Cuda Mem Copy Device to Host :" <<std::chrono::duration_cast<std::chrono::milliseconds>(end-afterKernels).count() <<" ms"<<endl;
	cout << "Total Time :" <<std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() <<" ms"<<endl;
	//free the memory!
	free(points);
	free(clusters);
    //CUDA FREE
    cudaFree( device_points );
    cudaFree( device_clusters );    
	return 0;
}