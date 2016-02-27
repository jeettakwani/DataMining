__author__ = 'jtakwani'
import sys
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import Counter
from operator import itemgetter
from numpy import arange, pi, cos, sin
#from scipy.stats import multivariate_normal
from numpy.random import rand

def main(arguments):
    filename = arguments[0]
    path = '/Users/jtakwani/PycharmProjects/DataMining/data/'+str(filename)

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
    cl = kmean(data,k)
    clusters = em(data,cl)
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
    return clusters

def em(data,clusters):
    new_likelihood = 0.0
    bo = True
    count = 0
    while(bo):
        mean_x = []
        mean_y = []
        alpha = []
        covariance = []
        mean_x_y = []
        likelihood = new_likelihood
        new_likelihood= 0.0
        cl = [[] for cl in range(len(clusters))]
        for point in data:
            probablities = []
            sum = 0.0
            for i in range(len(clusters)):
                X = [p[0] for p in clusters[i]]
                Y = [p[1] for p in clusters[i]]

                mean_x.append(np.mean(X))
                mean_y.append(np.mean(Y))
                mean_x_y.append((mean_x[i],mean_y[i]))
                alpha.append(float(len(clusters[i]))/float(len(data)))
                std_x = np.std(X)
                std_y = np.std(Y)
                std_matrix = np.matrix(np.vstack((std_x,std_y)))
                identity_matrix = np.matrix(np.identity(2))
                identity_matrix = 0.01*identity_matrix
                covariance.append(std_matrix*std_matrix.transpose()+identity_matrix)
                determinant_covariance = np.linalg.det(covariance[i])
                prior = 1/(2*pi*(math.sqrt(determinant_covariance)))
                term = np.matrix(np.hstack(([point[0]-mean_x[i]],[point[1]-mean_y[i]])))
                exponential_term = math.exp(-
                    (term*(np.linalg.inv(np.matrix(covariance[i])))*term.transpose())/2)
                sum += (alpha[i]*prior*exponential_term)
                probablities.append(alpha[i]*prior*exponential_term)

            index = 0
            max = probablities[0]
            for i in range(len(probablities)):
                if (max < probablities[i]):
                    max = probablities[i]
                    index = i
            cl[index].append(point)

            new_likelihood = new_likelihood + math.log(sum,2)
        clusters = cl
        if(math.fabs(likelihood-new_likelihood)<0.0001):
            bo = False
            break
        count = count+1
        print count
    print_covariance_mean_for_clusters(covariance,mean_x_y)
    return clusters

def print_covariance_mean_for_clusters(covariance,mean_x_y):
    fwrite = open("gmm_result.txt", "w")
    for i in range(3):
        fwrite.write("cluster :" + str(i))
        fwrite.write("\t")
        fwrite.write("covariance :" + str(covariance[i]))
        fwrite.write("\t")
        fwrite.write("mean :" + str(mean_x_y[i]))
        fwrite.write("\t")
def calculate_probablity(mean,alpha,covariance,data):
    clusters = [[] for i in range(3)]
    for point in data:
        probablities = []
        clusters = [[] for i in range(3)]
        for i in range(len(alpha)):
            determinant_covariance = np.linalg.det(covariance[i])
            prior = 1/(2*pi*(math.sqrt(determinant_covariance)))
            exponential_term = math.exp(-
                (np.matrix((point-mean[i]))*(np.inv(np.matrix(covariance[i])))*np.matrix((point-mean[i])))/2)
            probablities.append(alpha[i]*prior*exponential_term)
        for i in range(len(probablities)):
            max = probablities[0]
            index = 0
            if (max > probablities[i]):
                max = probablities[i]
                index = i
            clusters[index] = point

    return clusters

def get_covariance_matrix(sd):
    sd1 = np.matrix(sd)
    sd1.reshape(3,1)
    sd2 = np.matrix(sd)
    sd2.reshape(3,1)
    return sd1*sd2

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