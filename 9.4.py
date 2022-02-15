# Copyright (C) 2021 #
# @Time    : 2022/2/15 11:53
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : 8.3.py
# @Software: PyCharm


import numpy as np
import  matplotlib.pyplot as plt
def data_preprocessing():
    data = [[0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1], [0.608, 0.318, 1], [0.556, 0.215, 1],
            [0.403, 0.237, 1], [0.481, 0.149, 1], [0.437, 0.211, 1],
            [0.666, 0.091, 0], [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0], [0.639, 0.161, 0],
            [0.657, 0.198, 0], [0.360, 0.370, 0], [0.593, 0.042, 0], [0.719, 0.103, 0]]
    data = np.array(data)
    feature = data[:,:-1]
    category = data[:,-1]
    return  feature,category

def Kmeas(feature,category,k):

    #初始化聚类中心
    num = feature.shape[0]
    dim = feature.shape[1]
    print(num)
    random = np.random.randint(0,num,size=(k,1))
    center = feature[random]
    print(random)
    center = center.reshape(k, dim)
    distance = []
    tolerance = 1e-4
    optimizer = False
    cluster = {}
    for iters in range(100):
        for i in range(k):
            cluster[i + 1] = []

        print(cluster)
        for j in range(num):
            distance = []
            for i in range(k):
                distance.append(np.linalg.norm(feature[j] - center[i]))
            min_d = np.argmin(distance)
            cluster[min_d + 1].append(j)

        new_center = np.zeros_like(center)
        # print(np.average(feature[cluster[1]], axis=0))
        for i in range(k):
            new_center[i] = np.average(feature[cluster[i + 1]], axis=0)
        print('sum', np.sum(np.linalg.norm(center - new_center)))
        if np.sum(np.linalg.norm(center - new_center)) < 1e-6:

            optimizer = True
        else:
            center = new_center
        if optimizer:
            break
        print("Iter:{} ======= Center:{}".format(iters+1,cluster))
    return cluster,center

if __name__ == '__main__':
    f,c = data_preprocessing()
    cluster, center = Kmeas(f,c,2)
    print(cluster)


    color = ['g','c','b']
    for i in range(2):
        for data in cluster[i+1]:
            print(data)
            plt.scatter(x=f[data,0],y=f[data,1],c=color[i])
    plt.show()