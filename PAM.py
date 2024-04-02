import numpy as np
import time
import pandas as pd

def euclidean_distance(x1, x2):
    """Compute the Euclidean distance between two points."""
    return np.linalg.norm(x1 - x2)

def total_cost(data, medoids, clusters):
    """Calculate the total cost (sum of distances) of the clustering."""
    cost = 0
    for medoid_idx in medoids:
        for j in clusters[medoid_idx]:
            cost += euclidean_distance(data[j], data[medoid_idx])
    return cost

def assign_clusters(data, medoids):
    """Assign data points to the nearest medoids."""
    clusters = {medoid_idx: [] for medoid_idx in medoids}
    for i, point in enumerate(data):
        closest_medoid = min(medoids, key=lambda medoid_idx: euclidean_distance(point, data[medoid_idx]))
        clusters[closest_medoid].append(i)
    return clusters

def pam(data, k, max_iterations=100):
    """Partitioning Around Medoids (PAM) algorithm."""
    l = len(data)
    # Initialize medoids: select k random data points as initial medoids
    medoids = np.random.choice(range(l), k, replace=False)
    clusters = assign_clusters(data, medoids)
    old_cost = total_cost(data, medoids, clusters)
    td = old_cost

    for _ in range(max_iterations):
        # Iterate over each non-medoid point and try swapping with each medoid
        for x in range(l):
            if x not in medoids:
                for m in range(k):
                    # Swap the current non-medoid point with the m-th medoid
                    new_medoids = medoids.copy()
                    new_medoids[m] = x
                    new_clusters = assign_clusters(data, new_medoids)
                    new_cost = total_cost(data, new_medoids, new_clusters)

                    # If the new clustering has lower cost, update medoids and clusters
                    if new_cost < old_cost:
                        medoids = new_medoids
                        clusters = new_clusters
                        old_cost = new_cost
                        td = old_cost
        # If no improvement, break the loop
        else:
            break

    return td, medoids, clusters

# Example usage
if __name__ == "__main__":
    file_name = "Haberman.csv"
    df = pd.read_csv(file_name, header=None)
    # np.random.seed(0)
    X = np.random.randn(100, 2)
    X = df.to_numpy()

    # # Number of clusters
    k = 2
    # X = np.array([[3, 4], [9, 10], [5, 6], [7, 8], [1, 2]])
    # k = 4
    start_time = time.time()
    td, medoids, clusters = pam(X, k)
    end_time = time.time()
    print(end_time-start_time)
    print("Medoids:", X[medoids])
    print(np.sqrt(td))
    # for medoid_idx, cluster_points in clusters.items():
    #     print(f"Cluster {X[medoid_idx]}: {X[cluster_points]}")
