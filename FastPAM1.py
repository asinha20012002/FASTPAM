import numpy as np
import pandas as pd
import time

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def total_cost(data, medoids, clusters):
    cost = 0
    for medoid_idx in medoids:
        for j in clusters[medoid_idx]:
            cost += euclidean_distance(data[j], data[medoid_idx])
    return cost

def assign_clusters(data, medoids):
    clusters = {medoid_idx: [] for medoid_idx in medoids}
    for i, point in enumerate(data):
        closest_medoid = min(medoids, key=lambda medoid_idx: euclidean_distance(point, data[medoid_idx]))
        clusters[closest_medoid].append(i)
    return clusters

# fastpam1
def pam(data, k, max_iterations=100):
    l = len(data)
    if k == 1:
        medoid = np.median(data, axis=0)
        medoid_idx = np.argmin(np.sum((data - medoid) ** 2, axis=1))
        return [medoid_idx], {medoid_idx: list(range(l))}
    else:
        medoids = np.random.choice(range(l), k, replace=False)
        # medoids = np.arange(k)
        # print(medoids)
        clusters = assign_clusters(data, medoids)
        old_cost = total_cost(data, medoids, clusters)
        td = old_cost

        # Main loop for FASTPAM1 algorithm
        for _ in range(max_iterations):
            clusters = assign_clusters(data, medoids)
            nearest_medoids = {}
            second_nearest_medoids = {}
            
            for i in range(l):
                nearest_medoids[i] = min(medoids, key=lambda medoid_idx: euclidean_distance(data[i], data[medoid_idx]))
                second_nearest_medoids[i] = sorted(medoids, key=lambda medoid_idx: euclidean_distance(data[i], data[medoid_idx]))[1]

            DelTDF = 0  # Change in total deviation
            mstar = -1
            xstar = -1
        
            for xj, _ in enumerate(data):
                if xj in medoids:
                    continue
                dj = euclidean_distance(data[nearest_medoids[xj]], data[xj])

                DelTD ={i: 0 for i in range(l)}
                
                for i in medoids:
                    DelTD[i] = -dj
                    
                for xo, point in enumerate(data):
                    if xo == xj:
                        continue
                    doj = euclidean_distance(data[xo], data[xj])
                    n = nearest_medoids[xo]
                    dn = euclidean_distance(data[n], data[xo])
                    ds = euclidean_distance(data[second_nearest_medoids[xo]], data[xo])
                    DelTD[n] += min(doj, ds) - dn
                    if doj < dn:
                        for i in medoids:
                            if i == n:
                                continue
                            DelTD[i] += doj - dn
                i = np.argmin(DelTD)  # Get index of minimum value in DelTD
                if DelTD[i] < DelTDF:
                    DelTDF = DelTD[i]
                    mstar = medoids[i]
                    xstar = xj
            if DelTDF >= 0:
                break
            medoids = np.append(np.delete(medoids, np.where(medoids == mstar)), xstar)
            td = td + DelTDF  

        return td, medoids, clusters



if __name__ == "__main__":
    file_name = "Haberman.csv"
    df = pd.read_csv(file_name, header=None)
    # ground_truth_labels = (pd.read_csv("labels/"+file_name, header=None)).values.tolist()
    # ground_truth_labels = [item for sublist in ground_truth_labels for item in sublist]
    # scaler = StandardScaler()

    # Example usage
    # Generate some random data points
    np.random.seed(0)
    X = np.random.randn(100, 2)
    
    X = df.to_numpy()

    # Number of clusters
    k = 2
    # Example usage
    # X = np.array([[3, 4], [9, 10], [5, 6], [7, 8], [1, 2]])
    # k = 4

    # Run PAM algorithm
    start_time = time.time()
    td, medoids, clusters = pam(X, k)
    end_time = time.time()
    
    print(end_time - start_time)

    # # Print results
    # print("Medoids:", X[medoids])
    # print(td)
    # for medoid_idx, cluster_points in clusters.items():
    #     print(f"Cluster {X[medoid_idx]}: {X[cluster_points]}")
        
    # Find the minimum total deviation among medoid configurations
    min_td, min_medoids, min_clusters = td, medoids, clusters
    for _ in range(10):  # Run the algorithm multiple times to find the configuration with the minimum total deviation
        td, medoids, clusters = pam(X, k)
        if td < min_td:
            min_td, min_medoids, min_clusters = td, medoids, clusters

    # Print results with minimum total deviation
    print("Medoids:", X[min_medoids])
    print(np.sqrt(min_td))
    # for medoid_idx, cluster_points in min_clusters.items():
    #     print(f"Cluster {X[medoid_idx]}: {X[cluster_points]}")
