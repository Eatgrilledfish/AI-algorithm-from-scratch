import numpy as np

class KmeansClustering:
    """
    K-means clustering implementation for predefined initial centroids and fixed iterations.
    
    Args:
        k (int): Number of clusters
        data_points (np.ndarray): Input dataset with shape (n_samples, n_features)
        centroids (np.ndarray): Initial cluster centers with shape (k, n_features)
        iteration (int): Number of iterations to perform

    Attributes:
        centroids (np.ndarray): Final cluster centers after training
    """
    def __init__(self, k, data_points, centroids, iteration):
        self.k = k
        self.centroids = np.array(centroids)  # Ensure numpy array type
        self.data_points = np.array(data_points)
        self.iteration = iteration

    @staticmethod
    def euclidean_distance(data_point, centroids):
        """
        Calculate Euclidean distances between a data point and all centroids
        
        Args:
            data_point (np.ndarray): Single data point with shape (n_features,)
            centroids (np.ndarray): Cluster centers with shape (k, n_features)

        Returns:
            np.ndarray: Distance array with shape (k,)
        """
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def fit(self):
        """
        Execute K-means clustering process

        Steps:
        1. Assign points to nearest centroids
        2. Update centroids based on cluster means
        3. Handle empty clusters by retaining previous centroids

        Returns:
            np.ndarray: Final cluster centers
        """
        for _ in range(self.iteration):
            clusters = [[] for _ in range(self.k)]
            
            # Assign data points to nearest clusters
            for data_point in self.data_points:
                distances = self.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                clusters[cluster_num].append(data_point)
            
            # Update centroids
            new_centroids = []
            for i in range(self.k):
                if len(clusters[i]) == 0:  # Handle empty clusters
                    new_centroids.append(self.centroids[i])
                else:
                    new_centroids.append(np.mean(clusters[i], axis=0))
            
            self.centroids = np.array(new_centroids)
        
        return self.centroids
    

class RandomKmeansClustering:
    """
    K-means clustering with random centroid initialization and early stopping.
    
    Args:
        k (int): Number of clusters
        data_points (np.ndarray): Input dataset with shape (n_samples, n_features)
        iteration (int): Max number of iterations to perform

    Attributes:
        centroids (np.ndarray): Final cluster centers after training
    """
    def __init__(self, k, data_points, iteration):
        self.k = k
        self.data_points = np.array(data_points)
        self.iteration = iteration
        self.centroids = self._initialize_centroids()  # Initialize here

    def _initialize_centroids(self):
        """Randomly initialize centroids within data range"""
        min_vals = np.amin(self.data_points, axis=0)
        max_vals = np.amax(self.data_points, axis=0)
        return np.random.uniform(min_vals, max_vals, 
                               size=(self.k, self.data_points.shape[1]))  # Fix shape

    @staticmethod
    def euclidean_distance(data_point, centroids):
        """
        Calculate Euclidean distances between a data point and all centroids
        
        Args:
            data_point (np.ndarray): Single data point (n_features,)
            centroids (np.ndarray): Cluster centers (k, n_features)

        Returns:
            np.ndarray: Distance array (k,)
        """
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def fit(self):
        """
        Execute K-means clustering process with early stopping

        Returns:
            np.ndarray: Final cluster centers
        """
        for _ in range(self.iteration):
            clusters = [[] for _ in range(self.k)]
            
            # Assign points to clusters
            for data_point in self.data_points:
                distances = self.euclidean_distance(data_point, self.centroids)
                clusters[np.argmin(distances)].append(data_point)
            
            # Update centroids
            new_centroids = []
            for i, cluster in enumerate(clusters):
                new_centroids.append(
                    self.centroids[i] if len(cluster) == 0 
                    else np.mean(cluster, axis=0)
                )
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids, atol=1e-4):
                break
                
            self.centroids = np.array(new_centroids)
            
        return self.centroids