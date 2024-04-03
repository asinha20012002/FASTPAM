# Import necessary libraries
import numpy as np
import pandas as pd
import time

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# Function to calculate total cost of clustering
def total_cost(data, medoids, clusters):
    cost = 0
    for medoid_idx in medoids:
        for j in clusters[medoid_idx]:
            cost += euclidean_distance(data[j], data[medoid_idx])
    return cost

# Function to assign data points to clusters
def assign_clusters(data, medoids):
    clusters = {medoid_idx: [] for medoid_idx in medoids}
    for i, point in enumerate(data):
        closest_medoid = min(medoids, key=lambda medoid_idx: euclidean_distance(point, data[medoid_idx]))
        clusters[closest_medoid].append(i)
    return clusters

# PAM (Partitioning Around Medoids) algorithm
def pam(data, k, max_iterations=1000):
    l = len(data)
    # If only one cluster is requested, return the whole dataset as a single cluster
    if k == 1:
        medoid = np.median(data, axis=0)
        medoid_idx = np.argmin(np.sum((data - medoid) ** 2, axis=1))
        return [medoid_idx], {medoid_idx: list(range(l))}
    else:
        # Initialize medoids randomly
        medoids = np.random.choice(range(l), k, replace=False)

        # Main loop for FASTPAM1 algorithm
        for _ in range(max_iterations):
            clusters = assign_clusters(data, medoids)
            for medoid in medoids:
                # Update each medoid to the point with the minimum total distance to other points in its cluster
                medoid_points = [data[j] for j in clusters[medoid]]
                medoid_point = np.median(medoid_points, axis=0)
                medoid = np.argmin(np.sum((medoid_points - medoid_point) ** 2, axis=1))
                
            old_cost = total_cost(data, medoids, clusters)
            td = old_cost
            nearest_medoids = {}
            second_nearest_medoids = {}
            
            # Pre-calculate nearest and second nearest medoids for each data point
            for i in range(l):
                arr = sorted(medoids, key=lambda medoid_idx: euclidean_distance(data[i], data[medoid_idx]))
                nearest_medoids[i] = arr[0]
                second_nearest_medoids[i] = arr[1]

            DelTDF = 0  # Change in total deviation
            mstar = -1
            xstar = -1
        
            # Iterate through each data point to find potential swaps
            for xj, _ in enumerate(data):
                if xj in medoids:
                    continue
                dj = euclidean_distance(data[nearest_medoids[xj]], data[xj])

                DelTD ={i: 0 for i in range(l)}
                
                # Calculate change in total deviation if xj is swapped with each medoid
                for i in medoids:
                    DelTD[i] = -dj
                    
                for xo, _ in enumerate(data):
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
                i = np.argmin(list(DelTD.values()))  # Get index of minimum value in DelTD
                if DelTD[i] < DelTDF:
                    DelTDF = DelTD[i]
                    mstar = medoids[i]
                    xstar = xj
            if DelTDF >= 0:
                break
            medoids = np.append(np.delete(medoids, np.where(medoids == mstar)), xstar)
            td = td + DelTDF 
            
         

        return td, medoids, clusters

# Main execution
if __name__ == "__main__":
    # Load dataset
    file_name = "Haberman.csv"
    df = pd.read_csv(file_name, header=None)
    X = df.to_numpy()

    # Number of clusters
    k = 2

    # Run PAM algorithm
    start_time = time.time()
    td, medoids, clusters = pam(X, k)
    end_time = time.time()
    
    print(end_time - start_time)
        
    # Find the minimum total deviation among medoid configurations
    min_td, min_medoids, min_clusters = td, medoids, clusters
    
    # Print results with minimum total deviation
    print(np.sqrt(min_td))
