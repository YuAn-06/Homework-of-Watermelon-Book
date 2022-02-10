# Copyright (C) 2021 #
# @Time    : 2022/2/10 11:05
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : 3.2.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1/(1+np.e**(-z))

def data_preprocessing():
    data = [[0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1], [0.608, 0.318, 1], [0.556, 0.215, 1],
            [0.403, 0.237, 1], [0.481, 0.149, 1], [0.437, 0.211, 1],
            [0.666, 0.091, 0], [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0], [0.639, 0.161, 0],
            [0.657, 0.198, 0], [0.360, 0.370, 0], [0.593, 0.042, 0], [0.719, 0.103, 0]]
    data = np.array(data)

    x = data[:,:-1]
    Y = data[:,-1]
    bias = np.ones((data.shape[0], 1))

    X = np.concatenate([x, bias], axis=1)

    return X,Y

def training(x,y):
    y = y.reshape(-1,1)
    num = x.shape[0]
    feature_num = x.shape[1]
    beta = np.ones((1,feature_num))
    z = np.dot(x,beta.T) #shape:(17,1)
    old_likehood = 0
    new_likehood = np.sum(-y*z + np.log(1+np.exp(z)))
    epoch = 0
    # Maximize the likelihood function
    while (np.abs(new_likehood-old_likehood)>1e-7):
        p1 = np.exp(z) / (1 + np.exp(z))  # shape:(17,1)
        first_order = -np.sum(x * (y - p1), axis=0, keepdims=True)  # shape: (1,3)
        print(first_order.shape)
        p = np.diag((p1 * (1 - p1)).reshape(num))

        second_order = x.T.dot(p).dot(x)  # shape:(3,3)

        beta -= first_order.dot(np.linalg.inv(second_order))


        old_likehood = new_likehood
        z = np.dot(x, beta.T)
        new_likehood = np.sum(-y * z + np.log(1 + np.exp(z)))
        epoch +=1
    print("epoch: ",epoch,"beta:",beta)
    return beta
if __name__ == '__main__':

   x,y  = data_preprocessing()
   beta = training(x,y)
   x_1 = np.sort(np.random.random(100))
   x_2 = (-beta[0,0]*x_1-beta[0,2])/beta[0,1]
   plt.plot(x_1,x_2)

   plt.scatter(x[:8,0],x[:8,1],marker='o',c='red',label='True')
   plt.scatter(x[8:,0], x[8:,1], marker='o',c='green' ,label='False')
   plt.legend()
   plt.show()
