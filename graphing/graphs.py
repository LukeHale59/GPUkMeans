import matplotlib.pyplot as plt
import numpy as np
import os

# Create a folder named 'graphs' to save the plot
if not os.path.exists('graphs'):
    os.makedirs('graphs')

#CLUSTER VS RUNTIME
clusters = [5,25,50,100,200,500,1000]

runtime = [67,45,39,38,62,137,262]


#Set the figure size and font size
plt.figure(figsize=(6, 4))
plt.rcParams.update({'font.size': 10})

plt.plot(clusters, runtime, ':^', color='green', markersize=8, linewidth=2)

# Add axis labels and title
plt.xlabel('Number of Clusters')
plt.ylabel('Execution Time (ms)')
plt.suptitle("Numer of Clusters Vs. Execution Time")

# Add gridlines and legend
plt.grid(False)
#plt.legend(loc='best')


plt.savefig('graphs/VaryNumClusters.png',dpi=600)

# ORGIRNAL VS OPTIMIZED
kmeans_versions = ["Castros Kmeans", "Castro Cuda Ready"]
kmeans_times = [28628, 2678]  # Milliseconds

# Create a bar chart
plt.figure(figsize=(6, 4))
plt.rcParams.update({'font.size': 10})
plt.bar(kmeans_versions, kmeans_times, color=['blue', 'red'])
plt.xlabel("KMeans Version")
plt.ylabel("Running Time (ms)")
plt.title("Comparison of KMeans Running Times")

# Show the chart
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('graphs/OrginalVsNew.png',dpi=600)

#CPU VS GPU 600_0000
kmeans_versions2 = ["CPU", "GPU"]
kmeans_times2 = [2678, 38]  # Milliseconds

# Create a bar chart
plt.figure(figsize=(6, 4))
plt.rcParams.update({'font.size': 10})
plt.bar(kmeans_versions2, kmeans_times2, color=['blue', 'red'])
plt.xlabel("KMeans Version")
plt.ylabel("Running Time (ms)")
plt.title("KMeans on CPU vs. GPU on 600,000 points")

# Show the chart
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('graphs/CPUVsGPU.png',dpi=600)

#CPU VS GPU 6_000
kmeans_versions2 = ["CPU", "GPU"]
kmeans_times2 = [189, 7]  # Milliseconds

# Create a bar chart
plt.figure(figsize=(6, 4))
plt.rcParams.update({'font.size': 10})
plt.bar(kmeans_versions2, kmeans_times2, color=['blue', 'red'])
plt.xlabel("KMeans Version")
plt.ylabel("Running Time (ms)")
plt.title("KMeans on CPU vs. GPU on 6,000 points")

# Show the chart
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('graphs/CPUVsGPUWine.png',dpi=600)

#DOUBLE VS FLOAT
kmeans_versions2 = ["Double", "Float"]
kmeans_times2 = [68, 38]  # Milliseconds

# Create a bar chart
plt.figure(figsize=(6, 4))
plt.rcParams.update({'font.size': 10})
plt.bar(kmeans_versions2, kmeans_times2, color=['blue', 'red'])
plt.xlabel("Data Type Used")
plt.ylabel("Running Time (ms)")
plt.title("KMeans on GPU Varying Floating Point Accuracy")

# Show the chart
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('graphs/doubleVsFloat.png',dpi=600)