# Copyright (C) 2021 #
# @Time    : 2022/2/10 17:30
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : 3.5.py
# @Software: PyCharm

# Linear Discrimiant Analysis(LDA)
import numpy as np

def data_preprocessing():
    data = [[0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1], [0.608, 0.318, 1], [0.556, 0.215, 1],
            [0.403, 0.237, 1], [0.481, 0.149, 1], [0.437, 0.211, 1],
            [0.666, 0.091, 0], [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0], [0.639, 0.161, 0],
            [0.657, 0.198, 0], [0.360, 0.370, 0], [0.593, 0.042, 0], [0.719, 0.103, 0]]
    data = np.array(data)
    positive = data[:8,:-1]
    negative = data[8:,:-1]
    return  positive,negative

if __name__ == '__main__':
    positive , negative = data_preprocessing()

    mean_positive = np.mean(positive,axis=0).reshape(-1,1)
    mean_negative = np.mean(negative,axis=0).reshape(-1,1)
    print(mean_positive.shape)
    cov_positive = np.cov(positive,rowvar=False) #shape:(2,2) the covairance between axis
    cov_negatvie = np.cov(negative,rowvar=False) #shape:(2,2)
    print(cov_positive)
    sw = cov_positive + cov_negatvie
    sb = (mean_positive-mean_negative).dot((mean_positive-mean_negative).T)
    print(sb)
    beta = np.linalg.inv(sw).dot(mean_positive-mean_negative)
    print(beta)
    