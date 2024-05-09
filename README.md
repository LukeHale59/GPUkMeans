## GPUkMeans

**GPU-accelerated K-Means Clustering**

This repository implements the K-Means clustering algorithm using CUDA for efficient GPU acceleration. It's inspired by the work at [https://github.com/marcoscastro/kmeans](https://github.com/marcoscastro/kmeans).

**Key Features:**

* Leverages CUDA for significant performance gains on compatible GPUs compared to CPU-based implementations.
* Provides clear separation of CPU and GPU code for better maintainability.
* Offers flexibility to run the algorithm on both CPU and GPU for comparison or resource-constrained scenarios.

**Getting Started:**

**Prerequisites:**

* A CUDA-enabled NVIDIA GPU
* CUDA Toolkit installed ([https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads))

**Running the Code:**

**CPU:**

```bash
./runCPU.sh
```

**GPU:**

```bash
./runGPU.sh
```

**Configuration:**

The scripts `runCPU.sh` and `runGPU.sh` accept command-line options for fine-tuning various aspects of the K-Means algorithm. Here's a breakdown of the available options:

| Option | Description | Default Value |
|---|---|---|
| `-k` | Number of clusters (K) | 100 |
| `-s` | Random seed for initialization | 0 |
| `-h` | help |  |

**Explanation:**

- `-k`: This option allows you to specify the desired number of clusters (K) for the K-Means algorithm. The default value is typically defined within the script itself.
- `-s`: This option sets the random seed used for initializing the K-means algorithm. This helps control the initial placement of centroids and can influence the final clustering results.

**Modifying Configuration:**

To utilize these options, execute the scripts with the desired flags and arguments, cating the dataset you want to use. For example, to run wine dataset on the CPU version with 10 clusters, a random seed of 123:

```bash
cat datasets/wine.txt |  ./obj32/runCPU.sh -k 10 -s 123
```