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

//struct to store the points
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

//struct to store the clusters
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
    
	PointStruct* points = (PointStruct*)malloc(total_points * sizeof(PointStruct));

	// Initialize each PointStruct object
	for(int i = 0; i < total_points; i++) {
	    // Initialize the PointStruct object with the size of values array
	    new (&points[i]) PointStruct(total_values);
	
	    // Fill in the values for each 'values' array
	    for(int j = 0; j < total_values; j++) {
	        double value;
	        cin >> value;
	        points[i].values[j] = value;
	    }
	}

	ClusterStruct* clusters = (ClusterStruct*)malloc(K * sizeof(ClusterStruct(total_values)));
	std::chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now();
    
	if(K > total_points)
		return -1;
	vector<int> prohibited_indexes;
	// choose K distinct values for the centers of the clusters
	for(int i = 0; i < K; i++)
	{
		new (&clusters[i]) ClusterStruct(total_values);
		while(true)
		{
			int index_point = rand() % total_points;
			if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
					index_point) == prohibited_indexes.end())
			{
				prohibited_indexes.push_back(index_point);
				points[index_point].id_cluster=i;
				clusters[i].numPoints = 0;
				for(size_t j = 0;j<total_values;j++){
					clusters[i].central_values[j] = points[index_point].values[j];
					clusters[i].central_values_sums[j]=0;
				}
				break;
			}
		}
	}
    std::chrono::high_resolution_clock::time_point end_phase1 = chrono::high_resolution_clock::now();
    
    for(int i = 0; i < total_points; i++)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;
		for(int j = 0; j < total_values; j++)
		{
			sum += pow(clusters[0].central_values[j] -
					   points[i].values[j], 2.0);
		}
		min_dist = sum;
		for(int m = 1; m < K; m++)
		{
			double dist;
			sum = 0.0;
			for(int j = 0; j < total_values; j++)
			{
				dist = clusters[m].central_values[j] -points[i].values[j];
				sum += dist * dist;
			}
    	    //remove the sqrt
			if(sum < min_dist)
			{
				min_dist = sum;
				id_cluster_center = m;
			}
		}
		points[i].id_cluster = id_cluster_center;
	}
	int iter = 2;
	while(true)
	{
		bool done = true;
		//auto loop1_start = chrono::high_resolution_clock::now();
		//very small differnce at about 80 clusters
		// tbb::parallel_for( tbb::blocked_range<int>(0, K), [&](tbb::blocked_range<int> r){
        // for(int i = r.begin(); i < r.end(); i++){
        //     clusters[i].clearCentralValueSum();
        //     clusters[i].setTotalPoints(0);
        // }});
		//TBB IS SLOWER FOR LOOP 1
		for(int i = 0; i < K; i++){
			for(size_t j = 0 ; j < total_values;j++){
				clusters[i].central_values_sums[j]=0;
			}
            clusters[i].numPoints = 0;
        }
		//loop_duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - loop1_start);
		//cout << "Loop 1 duration: " << loop_duration.count() << " microseconds" << endl;
		auto loop2_start = chrono::high_resolution_clock::now();
        for(int i = 0; i < total_points; i++){
            for(int j = 0; j < total_values; j++){
                clusters[points[i].id_cluster].central_values_sums[j] += points[i].values[j];
            }
            clusters[points[i].id_cluster].numPoints++;
        }
		//loop_duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - loop2_start);
		//cout << "Loop 2 duration: " << loop_duration.count() << " microseconds" << endl;
        // tbb::parallel_for( tbb::blocked_range<int>(0, K), [&](tbb::blocked_range<int> r){
		// for(int i = r.begin(); i < r.end(); i++)
		// {
		// 	int total_points_cluster = clusters[i].getTotalPoints();
        //     for(int j = 0; j < total_values; j++){
        //         double sum = clusters[i].getCentralValueSum(j);
        //         clusters[i].setCentralValue(j, sum / total_points_cluster);
        //     }
		// }});
		//auto loop3_start = chrono::high_resolution_clock::now();
		//TBB IS SLOWER FOR LOOP 3
		// tbb::parallel_for( tbb::blocked_range<int>(0, K), [&](tbb::blocked_range<int> r){
        // for(int i = r.begin(); i < r.end(); i++){
        //     int total_points_cluster = clusters[i].getTotalPoints();
        //     for(int j = 0; j < total_values; j++){
        //         double sum = clusters[i].getCentralValueSum(j);
        //         clusters[i].setCentralValue(j, sum / total_points_cluster);
        //     }
        // }});
        for(int i = 0; i < K; i++){
            int total_points_cluster = clusters[i].numPoints;
            for(int j = 0; j < total_values; j++){
                double sum = clusters[i].central_values_sums[j];
                clusters[i].central_values[j] = sum / total_points_cluster;
            }
        }
		//loop_duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - loop3_start);
		//cout << "Loop 3 duration: " << loop_duration.count() << " microseconds" << endl;
        // associates each point to the nearest center
		// for(int i = 0; i < total_points; i++)
		// {
		// 	int id_old_cluster = points[i].getCluster();
		// 	int id_nearest_center = getIDNearestCenter(points[i]);
		// 	if(id_old_cluster != id_nearest_center)
		// 	{
		// 		points[i].setCluster(id_nearest_center);
		// 		done = false;
		// 	}
		// }
		//auto loop4_start = chrono::high_resolution_clock::now();
		//TBB MUCH FASTER FOR THIS PART ABOUT 10x, for wine, 12 attributes, k = 80 about ~230 ms
        // tbb::parallel_for( tbb::blocked_range<int>(0, total_points), [&](tbb::blocked_range<int> r){
		// for(int i = r.begin(); i < r.end(); i++)
		// {
		// 	int id_old_cluster = points[i].getCluster();
		// 	int id_nearest_center = getIDNearestCenter(points[i]);
		// 	if(id_old_cluster != id_nearest_center)
		// 	{
		// 		points[i].setCluster(id_nearest_center);
		// 		done = false;
		// 	}
		// }});
		//distence_kernel<<<number_of_blocks, threads_per_block>>>(points,total_points, K, total_values, clusters,done);
		for(int i = 0; i < total_points; i++)
		{
			int id_old_cluster = points[i].id_cluster;
			double sum = 0.0, min_dist;
			int id_cluster_center = 0;
			for(int j = 0; j < total_values; j++)
			{
				sum += pow(clusters[0].central_values[j] -
						   points[i].values[j], 2.0);
			}
			min_dist = sum;
			for(int m = 1; m < K; m++)
			{
				double dist;
				sum = 0.0;
				for(int j = 0; j < total_values; j++)
				{
					dist = clusters[m].central_values[j] -points[i].values[j];
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
				points[i].id_cluster = id_cluster_center;
				done = false;
			}
		}
		// for(int i = 0; i < total_points; i++)
		// {
		// 	int id_old_cluster = points[i].getCluster();
		// 	int id_nearest_center = getIDNearestCenter(points[i]);
		// 	if(id_old_cluster != id_nearest_center)
		// 	{
		// 		points[i].setCluster(id_nearest_center);
		// 		done = false;
		// 	}
		// }
		//loop_duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - loop4_start);
		//cout << "Loop 4 duration: " << loop_duration.count() << " microseconds" << endl;
		if(done == true || iter >= max_iterations)
		{
			//cout << "Break in iteration " << iter << "\n\n";
			break;
		}
		iter++;
	}
    std::chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
	// shows elements of clusters
	for(int i = 0; i < K; i++)
	{
		int total_points_cluster =  clusters[i].numPoints;
		cout << "Cluster " << i + 1 << endl;
		cout << "total_points_cluster " << total_points_cluster << endl;
		cout << "Cluster values: ";
		for(int j = 0; j < total_values; j++)
			cout << clusters[i].central_values[j] << " ";
		cout << endl;
	}
	cerr << "Iterations: " << iter << endl;
	        //cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";
    cout <<"Time for Serial: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-end_phase1).count() << " ms" <<endl;
    //cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";
    
    //cout << "TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";

	return 0;
}