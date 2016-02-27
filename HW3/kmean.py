import sys
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import Counter
from operator import itemgetter
from numpy import arange, pi, cos, sin
from numpy.random import rand

def main(arguments):
    filename = arguments[0]
    path = 'data/'+str(filename)

    if filename == "dataset3.txt":
        k = 2
    else:
        k = 3

    data = []

    read = open(path)
    for line in read:
        parts = line.split()
        data.append([float(parts[0]),float(parts[1]),int(parts[2])])
    read.close()
    clusters = kmean(data,k)
    temp = []
    for i, clust in enumerate(clusters):
        for z in clust:
            temp.append(z + [i+1])

    #print temp
    plotClusters(clusters)
    calculatePurity(temp,k)
    nmi(temp,k)

def kmean(data,k):
    centroids = random.sample(data,k)
    clusters = [[] for cluster in range(k) ]
    count = 0
    while True:
        clusters = [[]for cluster in range(k)]
        for point in data:
            minimum_dist = calculate_distance(point, centroids[0])
            label = 0
            for i in range(1,k):
                new_distance = calculate_distance(point,centroids[i])
                if new_distance < minimum_dist:
                    minimum_dist = new_distance
                    label = i
            clusters[label].append(point)
        maximum = 0.0
        for i in range(k):
            change = calculate_change(clusters[i],centroids,i)
            maximum = max(maximum,change)
        if maximum < 0.0001:
            break
    em(ckusters)
    return clusters

def em(clusters):
	mean = []
	for i in range(len(clusters):
		mean [i] = get_new_clusters(clusters[i])
def calculate_distance(point, centroid):
    x = pow((point[0]-centroid[0]),2)
    y = pow((point[1]-centroid[1]),2)
    return math.sqrt(x+y)

def calculate_change(clusters, centroids, k):
    old_centorid = centroids[k]
    new_centroid = get_new_centroid(clusters)
    centroids[k] = new_centroid
    change = calculate_distance(old_centorid,new_centroid)
    return change

def get_new_centroid(clusters):
    num_points = len(clusters)
    x = 0.0
    y = 0.0
    for j in clusters:
        x += j[0]
        y += j[1]
    return [x/num_points, y/num_points]
    #return max(change,new_value)

def plotClusters(data):
    x=[]
    y=[]
    colors = iter(cm.rainbow(np.linspace(0, 1, 3)))
    # Create a Figure object.
    fig = plt.figure(figsize=(5, 4))
    # Create an Axes object.
    ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    for i,c in enumerate(data):
        for p in c:
            x = x + [p[0]]
            y = y + [p[1]]
        ax.scatter(x,y,color=next(colors))
        x[:]=[]
        y[:]=[]
    ax.set_title("Kmeans Algorithm")
    plt.show()
    fig.savefig("scatterplot.png")

def calculatePurity(points,k):
    sum = 0.0
    for i in range(k):
        x = []
        for point in points:
            if point[3] == i+1:
                x.append(point)
        count = Counter(X[2] for X in x)
        count = count.items()
        sum += max(count,key=itemgetter(1))[1]

    print sum/len(points)
    return sum/len(points)

def nmi(points,k):
    sum = 0.0
    len_points = len(points)
    for i in range(k):
        class_label = []
        for point in points:
            if point[3] == i+1:
                class_label.append(point)
        cj = float(len(class_label))
        for j in range(k):
            ground_truth = []
            gt = []
            for point in class_label:
                if point[2] == j+1:
                    ground_truth.append(point)
            for point in points:
                if point[2] == j+1:
                    gt.append(point)
            wk = float(len(gt))
            wk_cj = float(len(ground_truth))
            if wk*cj == 0:
                continue
            if (len_points*wk_cj)/wk*cj == 0:
                continue
            sum += wk_cj*math.log(((len_points*wk_cj)/(wk*cj)),2)

    print sum/len_points
    return sum/len_points


if __name__ == '__main__':
    main(sys.argv[1:])