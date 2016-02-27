from pylab import *
from collections import Counter
from operator import itemgetter

class DbScanner:
    clusters = []
    visited = []
    data_set = []

    def calculate_epsilon(self):
        result = []
        sum = 0.0

        for point in self.data_set:
            distance = []
            for next_point in self.data_set:
                if point == next_point:
                    continue
                dist = math.sqrt(math.pow((point[0] - next_point[0]),2) +math.pow((point[1] - next_point[1]),2))
                distance.append(dist)
            distance.sort()
            result.append(distance[3])

        for r in result:
            sum += r
        print sum/len(result)
        return sum/float(len(result))

    def dbScan(self,data,Minpts,filename):
        self.data_set = data
        count = 1
        xlabel('x')
        ylabel('y')
        if filename == "dataset3.txt":
    		epsi = 0.13
    	else:
    		eps = self.calculate_epsilon()

        noise = Cluster(0)

        for data_point in self.data_set:
            if data_point not in self.visited:
                self.visited.append(data_point)
                neighborPoints = self.region_query(data_point, eps)
                #print len(neighborPoints)
                if len(neighborPoints) < Minpts:
                    noise.add(data_point)
                else:
                    clustr = Cluster(count)
                    clustr.add(data_point)

                    for point in neighborPoints:
                        if point not in self.visited:
                            self.visited.append(point)
                            neighbrPoints = self.region_query(point,eps)
                            if len(neighbrPoints) >= Minpts:
                                for p in neighbrPoints:
                                    if p not in neighborPoints:
                                        neighborPoints.append(p)

                        for c in self.clusters:
                            if not c.has_point(point):
                                if not clustr.has_point(point):
                                    clustr.add(point)
                        if len(self.clusters) == 0:
                            if not clustr.has_point(point):
                                clustr.add(point)
                    self.clusters.append(clustr)

                    count+=1
                    plot(clustr.get_x_coordinate(),clustr.get_y_coordinate(),'o')
                    hold(True)

        if len(noise.get_points())!= 0:
            plot(noise.get_x_coordinate(),noise.get_y_coordinate(),'*')

        temp = []
        i = 1
        for clst in self.clusters:
            temp = temp + clst.addlabel(i)
            i+=1

        self.calculatePurity(temp)
        self.nmi(temp)
        show()
        print len(self.clusters)

    def region_query(self, data_point, epsilon):
        result = []

        for point in self.data_set:

            if (math.sqrt(math.pow((point[0] - data_point[0]),2))+ math.pow((point[1] - data_point[1]),2)) <= epsilon:

                result.append(point)

        return result

    def calculatePurity(self,points):
        sum = 0.0
        for i in range(len(self.clusters)):
            x = []
            for point in points:
                if point[3] == i+1:
                    x.append(point)
            count = Counter(X[2] for X in x)
            count = count.items()
            sum += max(count,key=itemgetter(1))[1]

        print sum/len(points)
    #return sum/len(points)

    def nmi(self,points):
        sum = 0.0
        ground_label = []
        for p in points:
            if p[2] not in ground_label:
                ground_label.append(p[2])
        length_gl = len(ground_label)

        len_points = len(points)
        for i in range(len(self.clusters)):
            class_label = []
            for point in points:
                if point[3] == i+1:
                    class_label.append(point)
            cj = float(len(class_label))

            for j in range(length_gl):
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

class Cluster:
    points = []
    x_coordinate = []
    y_coordinate = []

    def __init__(self,name):
        self.name = name
        self.points = []
        self.x_coordinate = []
        self.y_coordinate = []

    def add(self,point):
        self.points.append(point)
        self.x_coordinate.append(point[0])
        self.y_coordinate.append(point[1])

    def get_x_coordinate(self):
        return self.x_coordinate

    def get_y_coordinate(self):
        return self.y_coordinate

    def has_point(self,point):
        if point not in self.points:
            return False
        return True

    def get_points(self):
        return self.points

    def addlabel(self,label):
        temp = []
        for p in self.points:
            p = p + [label]
            temp.append(p)
        return temp


def main(arguments):
    filename = arguments[0]
    path = 'data/'+str(filename)
    
    Minpts = 3
    data = []

    read = open(path)
    for line in read:
        parts = line.split()
        data.append([float(parts[0]),float(parts[1]),int(parts[2])])
    read.close()

    dbc = DbScanner()
    dbc.dbScan(data,Minpts,filename)


if __name__ == '__main__':
    main(sys.argv[1:])
