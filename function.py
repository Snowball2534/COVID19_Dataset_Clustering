import copy
import math
import numpy as np


# Euclidean distance
def euclidean(a,b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(len(a))))


# single linkage distance
# the shortest distance of two data points in the two clusters
def sld(d_dict, cluster1, cluster2):
    res = float('inf')
    for c1 in cluster1:
        for c2 in cluster2:
            dist = euclidean(d_dict[c1], d_dict[c2])
            if dist < res:
                res = dist
    return res

# complete linkage distance
# the longest distance of two data points in the two clusters
def cld(d_dict, cluster1, cluster2):
    res = 0.0
    for c1 in cluster1:
        for c2 in cluster2:
            dist = euclidean(d_dict[c1], d_dict[c2])
            if dist > res:
                res = dist
    return res


def hier_cluster(d_dict, k, op):
    '''
    Hierarchical clustering with single and complete linkages to cluster
    the states into k clusters based on their parameter values.
    Use Euclidean distance.

    Args:
        d_dict: The parameter values.
        k: The number of clusters to cluster.
        op: The flag to mark the distance to be used; 'False' marks the single
        linkage distance and 'True' marks the complete linkage distance.

    Returns:
        clusters: The clustering results.
    '''
    n = len(d_dict)
    clusters = [{d} for d in d_dict.keys()]  # initialize the clusters
    for _ in range(n - k):
        dist = float('inf')
        best_pair = (None, None)
        # search for two clusters to be merged in the next stage
        for i in range(len(clusters) - 1):
            for j in range(i + 1, len(clusters)):
                if op == False:  # single linkage distance
                    if sld(d_dict, clusters[i], clusters[j]) < dist:
                        dist = sld(d_dict, clusters[i], clusters[j])
                        best_pair = (i, j)
                else:  # complete linkage distance
                    if cld(d_dict, clusters[i], clusters[j]) < dist:
                        dist = cld(d_dict, clusters[i], clusters[j])
                        best_pair = (i, j)
        # update clusters
        new_clu = clusters[best_pair[0]] | clusters[best_pair[1]]
        clusters = [clusters[i] for i in range(len(clusters)) if i not in best_pair]
        clusters.append(new_clu)

    return clusters


# compute the center the cluster
def center(d_dict, cluster):
    return np.average([d_dict[c] for c in cluster], axis=0)

def kmeans(d_dict, k):
    '''
    K-means clustering to cluster the states into k clusters based on
    their parameter values.
    Use Euclidean distance.

    Args:
        d_dict: The parameter values.
        k: The number of clusters to cluster.

    Returns:
        clusters: The clustering results.
    '''
    states = list(d_dict)

    # randomly choose the initial centers from dataset
    init_num = np.random.choice(50 - 1, k)
    clusters = [{states[i]} for i in init_num]

    # update centers until convergence
    while True:
        new_clusters = [set() for _ in range(k)]
        centers = [center(d_dict, cluster) for cluster in clusters]
        for c in states:
            clu_ind = np.argmin([euclidean(d_dict[c], centers[i]) for i in range(k)])
            new_clusters[clu_ind].add(c)
        if all(new_clusters[i] == clusters[i] for i in range(k)):
            break
        else:
            clusters = copy.deepcopy(new_clusters)

    return clusters

def kmeans_analy(d_dict, clusters):
    # compute the center of each cluster
    cent = np.zeros((len(clusters), len(clusters)))
    i = 0
    for c in clusters:
        cent[i, :] = center(d_dict, c)
        i += 1
    print(cent)
    # compute the total distortion
    d = 0.0
    for i in range(len(clusters)):
        c = cent[i, :]
        for item in clusters[i]:
            d += euclidean(d_dict[item], c)**2
    print(d)