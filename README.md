# Spectral-Clustering
A Tutorial on Spectral Clustering - A simple example
Tutorial of Spectral Clustering:
Introduction: Clustering is a method of analyzing data that groups data to "maximize in-group similarity and minimize out-group similarity".Clustering is one of the unsupervised learning methods so it doesn’t use pre-defined labels for data. Probabilistic methods, distance-based methods, density-based methods, grid-based methods, factorization techniques, spectral methods are among the popular methods of clustering.
Spectral Clustering uses data spacing or similarity matrix instead of working with data itself or the main dimensions of the data. These methods can perform the task of placing data in Euclidean space while reducing their size. Hence, spectral methods are used to cluster arbitrary objects such as nodes in graphs. Spectral Clustering methods can be considered as a Graph-based technique that is used to cluster any type of data by converting them to a similarity matrix in a network structure such as a graph. When it comes to data forms with complicated shapes, Spectral Clustering is very useful. 
 
 

Algorithm:
In this algorithm we need a graph, so first we have to transform our dataset into a graph due to that we have to create a Similarity Matrix, and by constructing this matrix, our problem becomes a similarity graph where each interconnected component shows a cluster. In fact, in this graph, edges that are in the same cluster have a higher weight, and edges that are not in the same cluster have less weight. Then, calculate the Adjacency matrix ‘W’, Degree matrix ‘D’, and the Laplacian matrix ‘L’. We can use many distance measures for scoring the edges in L, but Euclidean Distance and Nearest Neighbors are more common.
After constructing Laplacian matrix we should compute the eigenvectors of it. Finally, with an algorithm such as K-Means, the desired clustering can be achieved from among the eigenvectors.
In the end, this algorithm also needs to get the expected number of clusters from the user, but in some cases and using the distance between eigenvectors, the optimal number can be selected for the number of clusters. This is called Rounding.
In a deeper look:
Similarity matrix: The input of the algorithm, a NxN matrix that N is the number of data points. the cells shows the distance (ex: Euclidean distance ) between each pair .
Adjacency matrix:  If a pair has connection by an edge so it has a weight in the adjacency matrix. For initializing it we copy the similarity matrix and select a threshold so if the distance score is above the threshold we consider it as 1 in adjacency matrix . It has a block diagonal form.
Degree matrix:  it is a diagonal matrix with degrees on the diagonal.
Laplacian matrix: Laplacian matrix is calculated by subtracting the adjacency matrix from the degree matrix:
L=D-W
After that we should normalize it.
Pseudocode:
Input: Similarity matrix S (nxn), 〖S∈R〗^(n×n) , and K the number of clusters.
	Create a similarity graph. 
	Create  Adjacency matrix W
	Compute the normalized Laplacian Matrix L.
	 Compute the first k eigenvectors of L , u_1,..., u_k .
	Normalize L 
	Let 〖U∈M〗^(n×k) be the matrix containing the vectors , u_1,..., u_k  as columns.
	For i=1,…,n let 〖y_i∈R〗^k be the vector corresponding to the i-th row of U.
	put the data points into the clusters by k-means algorithm.
	Place the centroids randomly 
	 Repeat steps 3 and 4 until convergence or until the end of a fixed number of iterations
	 for each data point x_i:
     find the nearest centroid(c_i,...,c_k) 
     assign the point to that cluster 
	 for each cluster j = 1..k
	       new centroid = mean of all points assigned to that cluster
	Output: data points in separated clusters.
Necessary libraries: sklearn, scipy, Numpy, seaborn, pandas, matplotlib.
There is a simple example to understand how it works : 
Pros, cons and improvement:

Despite of K-means algorithm, in spectral clustering consider the affinity is in addition of absolute location. So it would be more reliable and it can be more optimized.
We need to have a number of clusters as an input; however, after some iteration, we can find optimized number of clusters.(von Luxburg 2007, 395-416)
Schölkopf(Bernhard Schölkopf, John Platt, and Thomas Hofmann 2007) show that if the dataset contains different structures at scales of size and density the first few eigenvectors of L  matrix cannot put data points into the clusters successfully .








