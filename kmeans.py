import numpy as np
import pandas as pd

def kmeans(X, k):
    """
    K-Means
    Input:
        X - An (n, d) numpy array. Each row is a sample with d dimensions
        k - Number of clusters
    Output:
        centers   - A (k, d) numpy array. Each row is a center with d dimensions.
        clusters  - A (n,) numpy array indicating which cluster each sample belongs to.
    IMPORTANT note:
        Use the first k rows of X as the initial centers.
        Since the result depends on the initial centers, you will get
        wrong answers if you fail to do this.
    """
    # initial_centers = X[:k, :]
    # Do something...
    # return centers, clusters
    N = X.shape[0]  # num sample points
    d = X.shape[1]  # dimension of space

    #
    # INITIALIZATION PHASE
    # initialize centroids randomly as distinct elements of xs
    initial_centers = X[:k,:]
    centroids  = initial_centers
    assignments = np.zeros(N, dtype=np.uint8)

    # loop until convergence
    while True:
        # Compute distances from sample points to centroids
        # all  pair-wise _squared_ distances
        cdists = np.zeros((N, k))
        for i in range(N):
            xi = X[i, :]
            for c in range(k):
                cc  = centroids[c, :]
                #dist = 
                cdists[i, c] = np.linalg.norm(xi - cc)

        # Expectation step: assign clusters
        num_changed_assignments = 0
        for i in range(N):
            # pick closest cluster
            cmin = 0
            mindist = np.inf
            for c in range(k):
                if cdists[i, c] < mindist:
                    cmin = c
                    mindist = cdists[i, c]
            if assignments[i] != cmin:
                num_changed_assignments += 1
            assignments[i] = cmin

        # Maximization step: Update centroid for each cluster
        for c in range(k):
            points = X[assignments == c]
            #print(points.shape)
            temp = np.mean(points, axis=0)
            #print(temp.shape)
            centroids[c, :] = temp
            #print(new_center.shape)
            #newcent = 0
            #clustersize = 0
            #for i in range(N):
             #   if assignments[i] == c:
             #       newcent = newcent + X[i, :]
             #       clustersize += 1
            #newcent = newcent / clustersize
            #centroids[c, :]  = newcent

        if num_changed_assignments == 0:
            break

    # return cluster centroids and assignments
    return centroids, assignments
    #raise NotImplementedError


def sklearn_kmeans(X, k):
    """
    Your K-Means implementation should return clusters close to this function.
    Do not use this function in your implementation!
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, init=lambda x,n,random_state: x[:n, :], n_init=1, tol=1e-6)
    km.fit(X)
    return km.cluster_centers_, km.labels_


if __name__ == '__main__':
    n = 1000     # number of samples
    # n = 30000  # uncomment this line for an intense test case
    k = 10       # number of clusters
    data = pd.read_csv('data/adult.csv', header=None)
    data = data[[0, 2, 4]]
    X = data.to_numpy()
    X = X[:n]

    # normalize X
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X0 = X.copy()
    a, b = kmeans(X, k)
    a0, b0 = sklearn_kmeans(X0, k)

    print('Your clusters:', b)
    print('sklearn clusters:', b0)
    from sklearn.metrics import adjusted_rand_score
    print('ARI = ', adjusted_rand_score(b, b0))
