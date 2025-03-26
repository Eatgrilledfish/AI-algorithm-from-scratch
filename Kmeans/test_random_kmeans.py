import numpy as np
from Kmeans import RandomKmeansClustering

def test_basic_functionality():
    """Test clustering with obvious structure"""
    data = np.array([
        [1, 2], [1.5, 1.8], [1.2, 2.3],  # Cluster 1
        [5, 6], [6, 5.5], [5.5, 6.2]      # Cluster 2
    ])
    
    kmeans = RandomKmeansClustering(
        k=2,
        data_points=data,
        iteration=10
    )
    
    centers = kmeans.fit()
    
    # Verify cluster separation
    assert np.linalg.norm(centers[0] - centers[1]) > 3.0
    print("Basic functionality test passed!")

def test_dimension_handling():
    """Test 3D data handling"""
    data = np.random.rand(10, 3)  # 10 samples with 3 features
    kmeans = RandomKmeansClustering(
        k=3,
        data_points=data,
        iteration=5
    )
    
    centers = kmeans.fit()
    assert centers.shape == (3, 3)
    print("Dimension handling test passed!")

if __name__ == "__main__":
    test_basic_functionality()
    test_dimension_handling()