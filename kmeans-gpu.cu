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
    double* values;

	PointStruct() : id_cluster(-1), values(nullptr) {}

    // Constructor to initialize values array with a specified size
    PointStruct(int size) : id_cluster(-1) {
        values = new double[size];
    }

    // Destructor to free memory allocated for values array
    ~PointStruct() {
        delete[] values;
    }
};

struct ClusterStruct {
    int numPoints;
    double* central_values;
    double* central_values_sums;

	ClusterStruct() : numPoints(0), central_values(nullptr), central_values_sums(nullptr) {}
    // Constructor to initialize central_values and central_values_sums arrays with a specified size
    ClusterStruct(int size) : numPoints(0){
        numPoints = 0;
        central_values = new double[size];
        central_values_sums = new double[size];
    }

    // Destructor to free memory allocated for central_values and central_values_sums arrays
    ~ClusterStruct() {
        delete[] central_values;
        delete[] central_values_sums;
    }

};

__global__ void distence_kernel_first(double* points,double* clusters, int total_points, int total_values, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    //increment by stride
	for(int i = idx; i < total_points; i+= stride)
	{
			double sum = 0.0, min_dist;
			int id_cluster_center = 0;
			for(int j = 0; j < total_values; j++)
			{
				sum += pow(clusters[1+j] -
						   points[i*(total_values+1) +j + 1], 2.0);
			}
			min_dist = sum;
			for(int m = 1; m < K; m++)
			{
				double dist;
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

__global__ void clear_cluster_kernel(double* clusters, int total_values, int K)
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

__global__ void calculate_center_kernel(double* points,double* clusters, int total_points, int total_values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //int stride = gridDim.x * blockDim.x;
	//this needs to be fixed
	if(idx == 0){
    	for(int i = 0; i < total_points; i++){
			int clusterID = (int)points[i*(total_values+1)];
    	    for(int j = 0; j < total_values; j++){
				clusters[clusterID*(total_values*2 +1) +j + 1+ total_values] += points[i*(total_values+1) +j + 1];
    	   	}
    	clusters[clusterID * (total_values*2 +1)]++;
		}
	}
}

__global__ void get_center_kernel(double* clusters, int total_values, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    //increment by stride
			//clear each cluster value store
	for(int i = idx; i < K; i += stride){
		int total_points_cluster = clusters[i * (total_values*2 +1)];
        for(int j = 0; j < total_values; j++){
            double sum = clusters[i*(total_values*2 +1) +j + 1+ total_values];
            clusters[i*(total_values*2 +1) +j + 1] = sum / total_points_cluster;
        }
    }
}

__global__ void distence_kernel_main(double* points,double* clusters, int total_points, int total_values, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    //increment by stride
	for(int i = idx; i < total_points; i+= stride)
	{
		int id_old_cluster = points[i*(total_values+1)];
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;
		for(int j = 0; j < total_values; j++)
		{
			sum += pow(clusters[1+j] -
				   points[i*(total_values+1) +j + 1], 2.0);
		}
		min_dist = sum;
		for(int m = 1; m < K; m++)
		{
			double dist;
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
    
	//PointStruct* points = (PointStruct*)malloc(total_points * sizeof(PointStruct));

	double* points = (double*)malloc(sizeof(double) * total_points * total_values + sizeof(double) * total_points);

	// Initialize each PointStruct object
	for(int i = 0; i < total_points; i++) {
	    // Initialize the PointStruct object with the size of values array
	    // Fill in the values for each 'values' array
	    for(int j = 0; j < total_values; j++) {
	        double value;
	        cin >> value;
	        points[i*(total_values+1) +j + 1] = value;
	    }
	}

	//ClusterStruct* clusters = (ClusterStruct*)malloc(K * sizeof(ClusterStruct(total_values)));
	//values first, sums seconds
	double* clusters = (double*)malloc(sizeof(double) * K * total_values *2 + sizeof(double) * K);

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
	// std::chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now();

	//we now need to GPU

	double* device_points;
    cudaMalloc(&device_points, sizeof(double) * total_points * total_values + sizeof(double) * total_points);

    double* device_clusters;
    cudaMalloc(&device_clusters, sizeof(double) * K * total_values *2 + sizeof(double) * K);

    int threads_per_block = 512;
    int deviceId;
    cudaGetDevice(&deviceId);
  
    int numberOfSMs;
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    
    int number_of_blocks = 32 * numberOfSMs;
	cudaMemcpy(device_points, points, sizeof(double) * total_points * total_values + sizeof(double) * total_points, cudaMemcpyHostToDevice);
	cudaMemcpy(device_clusters, clusters, sizeof(double) * K * total_values *2 + sizeof(double) * K, cudaMemcpyHostToDevice);

	distence_kernel_first<<<number_of_blocks, threads_per_block>>>(device_points,device_clusters,total_points,total_values,  K);

	
    //assign each point to the correct cluster
	// for(int i = 0; i < total_points; i++)
	// {
	// 	double sum = 0.0, min_dist;
	// 	int id_cluster_center = 0;
	// 	for(int j = 0; j < total_values; j++)
	// 	{
	// 		sum += pow(clusters[1+j] -
	// 				   points[i*(total_values+1) +j + 1], 2.0);
	// 	}
	// 	min_dist = sum;
	// 	for(int m = 1; m < K; m++)
	// 	{
	// 		double dist;
	// 		sum = 0.0;
	// 		for(int j = 0; j < total_values; j++)
	// 		{
	// 			dist = clusters[m*(total_values*2 +1) +j + 1] -points[i*(total_values+1) +j + 1];
	// 			sum += dist * dist;
	// 		}
    // 	    //remove the sqrt
	// 		if(sum < min_dist)
	// 		{
	// 			min_dist = sum;
	// 			id_cluster_center = m;
	// 		}
	// 	}
	// 	points[i*(total_values+1)] = id_cluster_center;
	// }
	int iter = 2;
	while(true)
	{
		bool done = true;
		//clear each cluster value store
		// for(int i = 0; i < K; i++){
		// 	for(size_t j = 0 ; j < total_values;j++){
		// 		clusters[i*(total_values*2 +1) +j + 1+ total_values]=0;
		// 	}
        //     clusters[i * (total_values*2 +1)] = 0;
        // }

		clear_cluster_kernel<<<number_of_blocks, threads_per_block>>>(device_clusters,total_values,  K);

		//get the sum of all points in the cluster
        // for(int i = 0; i < total_points; i++){
		// 	int clusterID = (int)points[i*(total_values+1)];
        //     for(int j = 0; j < total_values; j++){
		// 		clusters[clusterID*(total_values*2 +1) +j + 1+ total_values] += points[i*(total_values+1) +j + 1];
        //     }
        //     clusters[clusterID * (total_values*2 +1)]++;
        // }

		calculate_center_kernel<<<number_of_blocks, threads_per_block>>>(device_points,device_clusters,total_points,total_values);
		//get the new centers for each cluster
        // for(int i = 0; i < K; i++){
        //     int total_points_cluster = clusters[i * (total_values*2 +1)];
        //     for(int j = 0; j < total_values; j++){
        //         double sum = clusters[i*(total_values*2 +1) +j + 1+ total_values];
        //         clusters[i*(total_values*2 +1) +j + 1] = sum / total_points_cluster;
        //     }
        // }
		get_center_kernel<<<number_of_blocks, threads_per_block>>>(device_clusters,total_values,  K);


		//assign each point to nearest cluster
		// for(int i = 0; i < total_points; i++)
		// {
		// 	int id_old_cluster = points[i*(total_values+1)];
		// 	double sum = 0.0, min_dist;
		// 	int id_cluster_center = 0;
		// 	for(int j = 0; j < total_values; j++)
		// 	{
		// 		sum += pow(clusters[1+j] -
		// 			   points[i*(total_values+1) +j + 1], 2.0);
		// 	}
		// 	min_dist = sum;
		// 	for(int m = 1; m < K; m++)
		// 	{
		// 		double dist;
		// 		sum = 0.0;
		// 		for(int j = 0; j < total_values; j++)
		// 		{
		// 			dist = clusters[m*(total_values*2 +1) +j + 1] -points[i*(total_values+1) +j + 1];
		// 			sum += dist * dist;
		// 		}
    	// 	    //remove the sqrt
		// 		if(sum < min_dist)
		// 		{
		// 			min_dist = sum;
		// 			id_cluster_center = m;
		// 		}
		// 	}
		// 	if(id_old_cluster != id_cluster_center)
		// 	{
		// 		points[i*(total_values+1)] = id_cluster_center;
		// 		done = false;
		// 	}
		// }

		distence_kernel_main<<<number_of_blocks, threads_per_block>>>(device_points,device_clusters,total_points,total_values,  K);
		
		if( iter == 10)
		{
			//cout << "Break in iteration " << iter << "\n\n";
			break;
		}
		iter++;
	}
	std::chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now();


	cudaMemcpy(points, device_points, sizeof(double) * total_points * total_values + sizeof(double) * total_points, cudaMemcpyDeviceToHost);
	cudaMemcpy(clusters, device_clusters, sizeof(double) * K * total_values *2 + sizeof(double) * K, cudaMemcpyDeviceToHost);

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
    cout <<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() <<endl;
        
	return 0;
}