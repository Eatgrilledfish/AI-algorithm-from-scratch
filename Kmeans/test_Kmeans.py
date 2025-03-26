import numpy as np
from Kmeans import KmeansClustering  # Assuming the class is saved in kmeans.py

def test_kmeans_basic():
    """Test basic K-means functionality with simple dataset"""
    # Test configuration
    data = np.array([
        [1.0, 2.0],    # Cluster 1 candidate
        [1.5, 1.8],     # Cluster 1 candidate
        [5.0, 8.0],     # Cluster 2 candidate 
        [8.0, 8.0],     # Cluster 2 candidate
        [1.0, 0.6],     # Cluster 1 candidate
        [9.0, 11.0]    # Cluster 2 candidate
    ])
    
    initial_centroids = np.array([
        [1.0, 2.0],    # Initial cluster 1 center
        [5.0, 8.0]      # Initial cluster 2 center
    ])
    
    # Initialize model
    kmeans = KmeansClustering(
        k=2,
        data_points=data,
        centroids=initial_centroids,
        iteration=3
    )
    
    # Execute clustering
    final_centroids = kmeans.fit()
    
    # Expected results (calculated manually)
    expected_cluster1_center = np.array([1.15, 1.35])    # Mean of [1.0,2.0], [1.5,1.8], [1.0,0.6]
    expected_cluster2_center = np.array([7.33, 9.0])     # Mean of [5.0,8.0], [8.0,8.0], [9.0,11.0]
    
    # Verify results with tolerance for floating point errors
    assert np.allclose(final_centroids[0], expected_cluster1_center, atol=0.1)
    assert np.allclose(final_centroids[1], expected_cluster2_center, atol=0.1)
    print("Basic test passed!")

def test_empty_cluster_handling():
    """Test cluster preservation when a cluster becomes empty"""
    # Test data where one cluster will lose all points
    data = np.array([
        [1.0, 1.0],
        [1.2, 0.8],
        [1.1, 1.1]
    ])  # All points clearly belong to one cluster
    
    # Initial centroids forced to create an empty cluster
    initial_centroids = np.array([
        [1.0, 1.0],    # Will capture all points
        [5.0, 5.0]      # Will lose all points
    ])
    
    kmeans = KmeansClustering(
        k=2,
        data_points=data,
        centroids=initial_centroids,
        iteration=2
    )
    
    final_centroids = kmeans.fit()
    
    # Verify empty cluster retains its original position
    assert np.array_equal(final_centroids[1], initial_centroids[1])
    print("Empty cluster test passed!")

if __name__ == "__main__":
    test_kmeans_basic()
    test_empty_cluster_handling()