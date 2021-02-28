'''
Assignment 2 - K-Means Clustering Algorithm
Dataset taken from UCI Machine Learning Repository
The code to plot has been commented out (scatter plots for k=3 attached)
The final centroids are indicated as 'x' on the plot

Team Members:
Sehajpreet Kaur (1020181129)
Aditi KP (1020181094)
Sejal Bansal (1020181445)
'''

import csv  # importing libraries
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt

# colors = itertools.cycle(["r", "b", "g", "c", "m", "y"])  # to plot all points in a cluster with one colour

with open('online_shoppers_intention.csv', newline='') as f:
    reader = csv.reader(f)
    data_t1 = list(reader)
data_t1 = np.array(data_t1)  # convert data to numpy array
data_t2 = data_t1[1:, 6:8]  # slice rows/columns to be taken
data = data_t2.astype(np.float)  # convert string to float for calculation

# plt.scatter(data[:, :1], data[:, 1:2], marker='.', c='black')  # plot the input data
# plt.title('Input dataset for performing clustering')
# plt.show()


def init_centroids(k, dataset):  # k = number of centroids
    arr = []
    n = np.random.randint(low=1, high=len(dataset), size=k)  # generating k random numbers to use as initial centroids
    while (len(n) != len(set(n))):  # making sure the centroids are unique
        n = np.random.randint(low=1, high=len(dataset), size=k)
    for i in n:  # for every i in n, get i th item from dataset and append to arr (it is a centroid)
        arr.append(dataset[i])
    return arr


def distance_table(centroids, dataset):
    distance_array = []
    for i in centroids:  # compute the distance between each centroid and all the points in the dataset
        temp = []  # temporary array to store the distances between points and that centroid
        for j in dataset:
            temp.append(calc_euc_distance(i, j))  # call function for euclidean distance
        distance_array.append(temp)  # append to array and move to next centroid
    return distance_array


def calc_euc_distance(p, q):  # function to calculate euclidean distance between two points
    sum = 0
    for i, j in zip(p, q):
        p = (i - j)*(i - j)
        sum = sum + p
    return math.sqrt(sum)


def clustering(k, dist_table):
    clusters = []  # empty list of clusters of size k
    for i in range(k):
        clusters.append([])
    for i in range(len(dist_table[0])):  # for each point in the distance table
        d = []  # temporary array to store the distances of all centroid from a point
        for j in range(len(dist_table)):
            d.append(dist_table[j][i])
        clusters[d.index(min(d))].append(i)  # cluster the point with the nearest centroid
    return clusters


def update_centroids(centroids, cluster_table, dataset):
    for i in range(len(centroids)):
        if len(cluster_table[i]) > 0:  # some items belong to this cluster (may have changed from previous iteration)
            temp = []
            for j in cluster_table[i]:
                temp.append(list(dataset[j]))  # copy current items in the cluster to a temporary list
            sum = [0] * len(centroids[i])
            for l in temp:  # find mean of all those items
                sum = [(a + b) for a, b in zip(sum, l)]
            centroids[i] = [p / len(temp) for p in sum]  # new centroid based on old and new items in this cluster
    return centroids


def check(past_dist, past_clusters):  # check if we have achieved perfect clusters and stop if necessary
    dist_const = all(x == past_dist[0] for x in past_dist)  # if distance table has remained constant
    cluster_const = all(y == past_clusters[0] for y in past_clusters)  # if cluster table has remained constant
    if dist_const:
        print("Stopping: No change in the distance table for the past few iterations")
    elif cluster_const:
        print("Stopping: No change in the cluster table for the past few iterations")
    return dist_const or cluster_const  # returns true if either have been met


def kMeans(k, dataset, max_iterations):
    past_dist = []  # to keep track of past distances as stopping criteria
    past_clusters = []  # to keep track of past clusters as stopping criteria
    centroids = init_centroids(k, dataset)  # initialise centroids randomly
    dist_table = distance_table(centroids, dataset)  # calculate distance table
    cluster_table = clustering(k, dist_table)  # cluster the points based on the distance table
    centroids_new = update_centroids(centroids, cluster_table, dataset)  # new centroids based on the cluster table
    past_dist.append(dist_table)
    past_clusters.append(cluster_table)

    for i in range(2, max_iterations):  # repeat steps till stopping criteria or max iterations met
        dist_table = distance_table(centroids_new, dataset)
        cluster_table = clustering(k, dist_table)
        centroids_new = update_centroids(centroids_new, cluster_table, dataset)
        past_dist.append(dist_table)
        past_clusters.append(cluster_table)
        if len(past_dist) > 3:  # stop if no change observed for more than 3 iterations
            past_dist.pop(0)  # only need to maintain data for past three iterations
            past_clusters.pop(0)
            if check(past_dist, past_clusters):
                print("Stopped at iteration number", i)
                break

    for i in range(len(centroids_new)):
        print("Centroid ", i+1, ": ", centroids_new[i])  # print the centroids
    #     color=next(colors)
    #     for j in range(len(cluster_table[i])):  # plot the data after clustering
    #         a = cluster_table[i][j]
    #         c = centroids_new[i]
    #         plt.scatter(data[a:a+1, 0:1], data[a:a+1, 1:2], marker='.', c=color)
    #         plt.scatter(c[0], c[1], marker='x', c='black')
    # plt.title('Output dataset after performing k-means clustering')
    # plt.show()


kMeans(3, data, 100)  # calling k-means with k=3, data as given and max iterations=100
