// Implementation of the KMeans Algorithm
// reference: https://github.com/marcoscastro/kmeans

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include "config_t.h"
#include <cassert>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <xmmintrin.h>


#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <xmmintrin.h>
#include <limits>
#include <cassert>
#include <cstdlib>
#include <string>
#include <unistd.h>


using namespace std;

class Point
{
private:
	int id_point, id_cluster;
	vector<double> values;
	int total_values;
	string name;

public:
	Point(int id_point, vector<double>& values, string name = "")
	{
		this->id_point = id_point;
		total_values = values.size();

		for(int i = 0; i < total_values; i++)
			this->values.push_back(values[i]);

		this->name = name;
		id_cluster = -1;
	}

	int getID()
	{
		return id_point;
	}

	void setCluster(int id_cluster)
	{
		this->id_cluster = id_cluster;
	}

	int getCluster()
	{
		return id_cluster;
	}

	double getValue(int index)
	{
		return values[index];
	}

	int getTotalValues()
	{
		return total_values;
	}

	void addValue(double value)
	{
		values.push_back(value);
	}

	string getName()
	{
		return name;
	}
};

class Cluster
{
private:
	int id_cluster;
	vector<double> central_values;
	vector<Point> points;

public:
	Cluster(int id_cluster, Point point)
	{
		this->id_cluster = id_cluster;

		int total_values = point.getTotalValues();

		for(int i = 0; i < total_values; i++)
			central_values.push_back(point.getValue(i));

		points.push_back(point);
	}

	void addPoint(Point point)
	{
		points.push_back(point);
	}

	bool removePoint(int id_point)
	{
		int total_points = points.size();

		for(int i = 0; i < total_points; i++)
		{
			if(points[i].getID() == id_point)
			{
				points.erase(points.begin() + i);
				return true;
			}
		}
		return false;
	}

	double getCentralValue(int index)
	{
		return central_values[index];
	}

	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

	Point getPoint(int index)
	{
		return points[index];
	}

	int getTotalPoints()
	{
		return points.size();
	}

	int getID()
	{
		return id_cluster;
	}
};

class KMeans
{
private:
	int K; // number of clusters
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;
	//no optimzation
	//return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(Point point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;

		for(int i = 0; i < total_values; i++)
		{
			sum += pow(clusters[0].getCentralValue(i) -
					   point.getValue(i), 2.0);
		}

		min_dist = sqrt(sum);

		for(int i = 1; i < K; i++)
		{
			double dist;
			sum = 0.0;

			for(int j = 0; j < total_values; j++)
			{
				sum += pow(clusters[i].getCentralValue(j) -
						   point.getValue(j), 2.0);
			}

			dist = sqrt(sum);

			if(dist < min_dist)
			{
				min_dist = dist;
				id_cluster_center = i;
			}
		}

		return id_cluster_center;
	}
	// return ID of nearest center (uses euclidean distance)
	//no sqrt, dist*dist
// int getIDNearestCenter(Point point)
// {
// 	double sum = 0.0, min_dist;
// 	int id_cluster_center = 0;
// 	for(int i = 0; i < total_values; i++)
// 	{
// 		sum += pow(clusters[0].getCentralValue(i) -
// 				   point.getValue(i), 2.0);
// 	}
// 	min_dist = sum;
// 	for(int i = 1; i < K; i++)
// 	{
// 		double dist;
// 		sum = 0.0;
// 		for(int j = 0; j < total_values; j++)
// 		{
// 			dist = clusters[i].getCentralValue(j) -point.getValue(j);
// 			sum += dist * dist;
// 		}
//         //remove the sqrt
// 		if(sum < min_dist)
// 		{
// 			min_dist = sum;
// 			id_cluster_center = i;
// 		}
// 	}
// 	return id_cluster_center;
// }


//  int getIDNearestCenter(Point point)
// {
//     double sum = 0.0, min_dist = numeric_limits<double>::max();
//     int id_cluster_center = -1;

//     // Compute the distance from the point to the remaining cluster centers
//     for(int i = 0; i < K; i++)
//     {
//         __m128d v_sum = _mm_setzero_pd();
//         for(int j = 0; j < total_values; j += 2)
//         {
//             __m128d v_point = _mm_set_pd(point.getValue(j+1), point.getValue(j));
// 			//could store the center of the cluster as a __m128d to speed up this part
//             __m128d v_center = _mm_set_pd(clusters[i].getCentralValue(j+1), clusters[i].getCentralValue(j));
//             __m128d v_diff = _mm_sub_pd(v_center, v_point);
//             __m128d v_squared_diff = _mm_mul_pd(v_diff, v_diff);
//             v_sum = _mm_add_pd(v_sum, v_squared_diff);
//         }

//         double distance[2];
//         _mm_store_pd(distance, v_sum);
//         sum = distance[0] + distance[1];

//         if(sum < min_dist)
//         {
//             min_dist = sum;
//             id_cluster_center = i;
//         }
//     }

//     return id_cluster_center;
// }
// int getIDNearestCenter(Point point)
// {
//     double sum = 0.0, min_dist = numeric_limits<double>::max();
//     int id_cluster_center = -1;

//     // Compute the distance from the point to the remaining cluster centers
//     for(int i = 0; i < K; i++)
//     {
//         __m256d v_sum = _mm256_setzero_pd();
// 		sum = 0;
//         // Process values in multiples of 4
//         for(int j = 0; j < total_values - (total_values % 4); j += 4)
//         {
//             __m256d v_point = _mm256_set_pd(point.getValue(j+3), point.getValue(j+2), point.getValue(j+1), point.getValue(j));
//             __m256d v_center = _mm256_set_pd(clusters[i].getCentralValue(j+3), clusters[i].getCentralValue(j+2), clusters[i].getCentralValue(j+1), clusters[i].getCentralValue(j));
//             __m256d v_diff = _mm256_sub_pd(v_center, v_point);
//             __m256d v_squared_diff = _mm256_mul_pd(v_diff, v_diff);
//             v_sum = _mm256_add_pd(v_sum, v_squared_diff);
//         }
// 			double dist;
//         	for(int j = total_values - (total_values % 4); j < total_values; j++)
//         	{
//         	    dist = clusters[i].getCentralValue(j) -point.getValue(j);
//         	    sum += dist * dist;
//         	}
//         double distance[4];
//         _mm256_store_pd(distance, v_sum);
//         sum += distance[0] + distance[1] + distance[2] + distance[3];

//         if(sum < min_dist)
//         {
//             min_dist = sum;
//             id_cluster_center = i;
//         }
//     }

//     return id_cluster_center;
// }

public:
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

	void run(vector<Point> & points)
	{
        std::chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now();
        
		if(K > total_points)
			return;

		vector<int> prohibited_indexes;

		// choose K distinct values for the centers of the clusters
		for(int i = 0; i < K; i++)
		{
			while(true)
			{
				int index_point = rand() % total_points;

				if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
						index_point) == prohibited_indexes.end())
				{
					prohibited_indexes.push_back(index_point);
					points[index_point].setCluster(i);
					Cluster cluster(i, points[index_point]);
					clusters.push_back(cluster);
					break;
				}
			}
		}
		// for(int i = 0; i < K; i++)
		// {
		// 	int total_points_cluster =  clusters[i].getTotalPoints();

		// 	cout << "Cluster " << clusters[i].getID() + 1 << endl;
		// 	cout << "total_points_cluster " << total_points_cluster << endl;
		// 	cout << "Cluster values: ";

		// 	for(int j = 0; j < total_values; j++)
		// 		cout << clusters[i].getCentralValue(j) << " ";
		// 	cout << endl;
		// }
        std::chrono::high_resolution_clock::time_point end_phase1 = chrono::high_resolution_clock::now();
        
		int iter = 1;

		while(true)
		{
			bool done = true;

			// associates each point to the nearest center
			for(int i = 0; i < total_points; i++)
			{
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = getIDNearestCenter(points[i]);

				if(id_old_cluster != id_nearest_center)
				{
					if(id_old_cluster != -1)
						clusters[id_old_cluster].removePoint(points[i].getID());

					points[i].setCluster(id_nearest_center);
					clusters[id_nearest_center].addPoint(points[i]);
					done = false;
				}
			}

			// recalculating the center of each cluster
			for(int i = 0; i < K; i++)
			{
				for(int j = 0; j < total_values; j++)
				{
					int total_points_cluster = clusters[i].getTotalPoints();
					double sum = 0.0;

					if(total_points_cluster > 0)
					{
						for(int p = 0; p < total_points_cluster; p++)
							sum += clusters[i].getPoint(p).getValue(j);
						clusters[i].setCentralValue(j, sum / total_points_cluster);
					}
				}
			}

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
			int total_points_cluster =  clusters[i].getTotalPoints();

			cout << "Cluster " << clusters[i].getID() + 1 << endl;
			// for(int j = 0; j < total_points_cluster; j++)
			// {
			// 	cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
			// 	for(int p = 0; p < total_values; p++)
			// 		cout << clusters[i].getPoint(j).getValue(p) << " ";

			// 	string point_name = clusters[i].getPoint(j).getName();

			// 	if(point_name != "")
			// 		cout << "- " << point_name;

			// 	cout << endl;
			// }
			cout << "total_points_cluster " << total_points_cluster << endl;
			cout << "Cluster values: ";

			for(int j = 0; j < total_values; j++)
				cout << clusters[i].getCentralValue(j) << " ";
			cout << endl;
			// //cout << "\n\n";
            // //cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";
            
            // //cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";
            
            // //cout << "TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";
		}
		            //cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";
            cout <<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() <<endl;
            //cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";
            
            //cout << "TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";
	}
};

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

	

	vector<Point> points;
	string point_name;

	for(int i = 0; i < total_points; i++)
	{
		vector<double> values;

		for(int j = 0; j < total_values; j++)
		{
			double value;
			cin >> value;
			values.push_back(value);
		}

		if(has_name)
		{
			cin >> point_name;
			Point p(i, values, point_name);
			points.push_back(p);
		}
		else
		{
			Point p(i, values);
			points.push_back(p);
		}
	}

	KMeans kmeans(K, total_points, total_values, max_iterations);
	kmeans.run(points);

	return 0;
}