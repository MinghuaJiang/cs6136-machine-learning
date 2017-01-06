#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# Your code here


def loadData(fileDj):
    data = np.loadtxt(fileDj)
    np.random.shuffle(data)
    y = data[:, len(data[0]) - 1]
    x = data[:, :len(data[0]) - 1]
    return x, y


## K-means functions

def getInitialCentroids(X, k):
    length = len(X[0])
    initialCentroids = {}
    for i in range(1, k + 1):
        centroid = np.zeros(length)
        for j in range(0, length):
            min_val = min(X[:, j])
            max_val = max(X[:, j])
            centroid[j] = np.random.randint(min_val, max_val)
        initialCentroids[i] = {}
        initialCentroids[i]["centroid"] = centroid

    return initialCentroids


def getDistance(pt1, pt2):
    # dist = np.linalg.norm(pt1 - pt2)
    dist = np.sqrt(np.sum(np.square(pt1 - pt2)))
    return dist


def allocatePoints(X, clusters):
    # Your code here
    rows = len(X)
    for key in clusters.keys():
        clusters[key]["cluster"] = []
    for i in range(0, rows):
        pt1 = X[i]
        min_distance = sys.maxint
        cluster_key = -1
        for key in clusters.keys():
            centroid_pt = clusters[key]["centroid"]
            distance = getDistance(pt1, centroid_pt)
            if distance < min_distance:
                min_distance = distance
                cluster_key = key
        clusters[cluster_key]["cluster"].append(pt1)

    return clusters


def updateCentroids(clusters):
    is_stable = True
    for key in clusters.keys():
        updated_centroid = np.mean(np.array(clusters[key]["cluster"]), axis=0)
        if not np.array_equal(updated_centroid, clusters[key]["centroid"]):
            clusters[key]["centroid"] = updated_centroid
            is_stable = False
    return clusters, is_stable


def visualizeClusters(clusters):
    color = {1: 'r', 2: 'g'}
    for i in clusters.keys():
        cluster = clusters[i]["cluster"]
        for each in cluster:
            plt.scatter(each[0], each[1], color=color[i])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Q3')
    plt.show()
    # Your code here


def kmeans(X, k, maxIter=1000):
    clusters = getInitialCentroids(X, k)
    for i in range(0, maxIter):
        clusters = allocatePoints(X, clusters)
        clusters, is_stable = updateCentroids(clusters)
        if is_stable:
            break
    clusters = allocatePoints(X, clusters)
    return clusters


def kneeFinding(X, kList):
    objectiveList = []
    for k in kList:
        clusters = kmeans(X, k)
        objective = 0.0
        for i in clusters.keys():
            centroid = clusters[i]["centroid"]
            cluster = clusters[i]["cluster"]
            for cluster_point in cluster:
                objective += np.sum(np.square(cluster_point - centroid))
        objectiveList.append(objective)
    plt.plot(kList, objectiveList)
    plt.xlabel("k")
    plt.ylabel("objective function")
    plt.title("Q4")
    plt.show()


def purity(X, true_labels, clusters):
    label_dict = dict()
    for i in range(0, len(X)):
        data = X[i]
        label = true_labels[i]
        label_dict[tuple(data)] = label

    purities = []
    for key in clusters.keys():
        counter = Counter()
        cluster = clusters[key]["cluster"]
        n = len(cluster)
        for each in cluster:
            counter[label_dict[tuple(each)]] += 1
        purities.append(float(counter.most_common(1)[0][1]) / n)

    return purities


## GMM functions

# calculate the initial covariance matrix
# covType: diag, full
def getInitialsGMM(X, k, covType):
    if covType == 'full':
        dataArray = np.transpose(X)
        covMat = np.cov(dataArray)

    else:
        covMatList = []
        for i in range(len(X[0])):
            data = X[:, i]
            cov = np.cov(data)
            covMatList.append(cov)
        covMat = np.diag(covMatList)

    initialClusters = {}

    length = len(X[0])

    splitted_x = np.array_split(X, k)
    for i in range(1, k + 1):
        initialClusters[i] = {}
        initialClusters[i]["mean"] = np.mean(splitted_x[i - 1], axis=0)
        initialClusters[i]["cov"] = covMat
        initialClusters[i]["covinv"] = np.linalg.inv(covMat)
        initialClusters[i]["covdet"] = np.linalg.det(covMat)
        initialClusters[i]["prop"] = 1.0 / k
        initialClusters[i]["cluster"] = []
    # print(initialClusters)
    return initialClusters


def calcLogLikelihood(X, clusters):
    loglikelihood = 0.0
    for i in range(0, len(X)):
        for key in clusters.keys():
            loglikelihood += np.log(np.exp(
                -0.5 * np.dot(
                    np.dot(np.transpose(X[i] - clusters[key]["mean"]), clusters[key]["covinv"]),
                    X[i] - clusters[key]["mean"])) * clusters[key]["prop"] / (
                     2 * np.pi * np.sqrt(clusters[key]["covdet"])))

    return loglikelihood


# E-step
def updateEStep(X, clusters, k):
    EMatrix = []
    for i in range(0, len(X)):
        row = np.zeros(k)
        total = 0.0
        for j in clusters.keys():
            total += np.exp(
                -0.5 * np.dot(
                    np.dot(np.transpose(X[i] - clusters[j]["mean"]), clusters[j]["covinv"]),
                    X[i] - clusters[j]["mean"])) * clusters[j]["prop"]

        for key in clusters.keys():
            e_row_key = np.exp(
                -0.5 * np.dot(np.dot(np.transpose(X[i] - clusters[key]["mean"]), clusters[key]["covinv"]),
                              X[i] - clusters[key]["mean"])) * clusters[key]["prop"]
            row[key - 1] = e_row_key / total
        EMatrix.append(row)
    return EMatrix


# M-step
def updateMStep(X, clusters, EMatrix):
    e_matrix = np.array(EMatrix)
    sum = np.sum(e_matrix, axis=0)
    is_stable = True
    for j in range(0, len(e_matrix[0])):
        j_sum = 0.0
        for i in range(0, len(X)):
            j_sum += e_matrix[i][j] * X[i]
        updated_mean = j_sum / sum[j]
        if getDistance(clusters[j + 1]["mean"], updated_mean) > 1e-3:
            is_stable = False
            clusters[j + 1]["mean"] = updated_mean
            clusters[j + 1]["prop"] = sum[j] / len(e_matrix)
    return clusters, is_stable


def visualizeClustersGMM(clusters, covType, problem):
    color = {1: 'r', 2: 'g'}
    for i in clusters.keys():
        cluster = clusters[i]["cluster"]
        for each in cluster:
            plt.scatter(each[0], each[1], color=color[i])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(problem + ':' + covType)
    plt.show()


def gmmCluster(X, k, covType, maxIter=1000):
    # initial clusters
    clustersGMM = getInitialsGMM(X, k, covType)
    for i in range(0, maxIter):
        e_matrix = updateEStep(X, clustersGMM, k)
        clustersGMM, is_stable = updateMStep(X, clustersGMM, e_matrix)
        if is_stable:
            break
    e_matrix = updateEStep(X, clustersGMM, k)

    for j in range(0, len(e_matrix)):
        key = np.argmax(e_matrix[j]) + 1
        clustersGMM[key]["cluster"].append(X[j])

    return clustersGMM


def purityGMM(X, true_label, clusters):
    return purity(X, true_label, clusters)


def main():
    #######dataset path
    datadir = sys.argv[1]
    pathDataset1 = datadir + '/humanData.txt'
    pathDataset2 = datadir + '/audioData.txt'
    np.random.seed(0)
    dataset1, true_label1 = loadData(pathDataset1)
    np.random.seed(15)
    dataset2, true_label2 = loadData(pathDataset2)

    # Q4
    kneeFinding(dataset1, range(1, 7))

    # Q2
    clusters = kmeans(dataset1, 2)
    # Q3
    visualizeClusters(clusters)
    # Q5
    purity_q5 = purity(dataset1, true_label1, clusters)
    print("purity for Q3: %s" % purity_q5)

    # Q7
    clustersGMM11 = gmmCluster(dataset1, 2, 'diag')
    visualizeClustersGMM(clustersGMM11, 'diag', 'Q7')
    clustersGMM12 = gmmCluster(dataset1, 2, 'full')
    visualizeClustersGMM(clustersGMM12, 'full', 'Q7')
    # Q8
    clustersGMM21 = gmmCluster(dataset2, 2, 'diag')
    visualizeClustersGMM(clustersGMM21, 'diag', 'Q8')
    clustersGMM22 = gmmCluster(dataset2, 2, 'full')

    # Q9
    purities11 = purityGMM(dataset1, true_label1, clustersGMM11)
    print("purity for Q7 diag: %s" % purities11)
    purities12 = purityGMM(dataset1, true_label1, clustersGMM12)
    print("purity for Q7 full: %s" % purities12)
    purities21 = purityGMM(dataset2, true_label2, clustersGMM21)
    print("purity for Q8 diag: %s" % purities21)
    purities22 = purityGMM(dataset2, true_label2, clustersGMM22)
    print("purity for Q8 full: %s" % purities22)


if __name__ == "__main__":
    main()
