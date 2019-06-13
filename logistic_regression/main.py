#The data
#我们将建立一个逻辑回归模型来预测一个学生是否被大学录取。假设你是一个大学系的管理员，你想根据两次考试的结果来决定每个申请人的录取机会。你有以前的申请人的历史数据，你可以用它作为逻辑回归的训练集。对于每一个培训例子，你有两个考试的申请人的分数和录取决定。为了做到这一点，我们将建立一个分类模型，根据考试成绩估计入学概率。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
path = 'data' + os.sep + 'LogiReg_data.txt'   #os.sep不同平台使用不同平台的分隔符
pdData = pd.read_csv(path, header=None, names=['Exams 1', 'Exams 2', 'Admitted'])   #读取csv文件,header自己指定names
print(pdData.head())
print(pdData.shape)

positive = pdData[pdData['Admitted'] == 1] #指定正例,把正例挑出来
negative = pdData[pdData['Admitted'] == 0]  #指定反例
#print(positive)

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(positive['Exams 1'], positive['Exams 2'], s=30, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exams 1'], negative['Exams 2'], s=30, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')

#sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

nums = np.arange(-10, 10, step=1) #creates a vector containing 20 equally spaced values from -10 to 10
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(nums, sigmoid(nums), 'r')
print('=======')


def model(X, theta):
    return sigmoid(np.dot(X, theta.T))

#添加一列便于计算
pdData.insert(0, 'Ones', 1) # in a try / except structure so as not to return an error if the block si executed several times
print('pdData.shape是：',pdData.shape)


# set X (training data) and y (target variable)
orig_data = pdData.as_matrix() # convert the Pandas representation of the data to an array useful for further computations
print('orig_data.shape是：',orig_data.shape)
cols = orig_data.shape[1]
X = orig_data[:,0:cols-1]
y = orig_data[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
#X = np.matrix(X.values)
#y = np.matrix(data.iloc[:,3:4].values) #np.array(y.values)
theta = np.zeros([1, 3])

print(X[:5])
print(y[:5])
print(theta)
print(X.shape, y.shape, theta.shape)

#损失函数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))

cost(X, y, theta)

#计算梯度
def gradient(X, y, theta):
    # 求出theta.shape个梯度，初始化为0
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel() #ravel()扁平化操作：https://www.cnblogs.com/mzct123/p/8659193.html
    for j in range(len(theta.ravel())):  # for each parmeter
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)

    return grad

#比较3中不同梯度下降方法
STOP_ITER = 0        #根据迭代次数，更新一次参数算一次迭代
STOP_COST = 1        #根据损失值的变化
STOP_GRAD = 2        #根据梯度变化

def stopCriterion(type, value, threshold):
    #设定三种不同的停止策略
    if type == STOP_ITER:        return value > threshold
    elif type == STOP_COST:      return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD:      return np.linalg.norm(value) < threshold    #np.linalg.norm求范数

import numpy.random
#洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y


import time


def descent(data, theta, batchSize, stopType, thresh, alpha):
    # 梯度下降求解

    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)  # 计算的梯度
    costs = [cost(X, y, theta)]  # 损失值

    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize  # 取batch数量个数据
        if k >= n:
            k = 0
            X, y = shuffleData(data)  # 重新洗牌
        theta = theta - alpha * grad  # 参数更新
        costs.append(cost(X, y, theta))  # 计算新的损失
        i += 1

        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType, value, thresh): break

    return theta, i - 1, costs, grad, time.time() - init_time



def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    #import pdb; pdb.set_trace();
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize==n: strDescType = "Gradient"
    elif batchSize==1:  strDescType = "Stochastic"
    else: strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    plt.show()
    return theta

#设定迭代次数
#选择的梯度下降方法是基于所有样本的
n=100   #选择整个数据集来进行梯度下降
#runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)

#设定阈值 1E-6, 差不多需要110 000次迭代
#runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)

#设定阈值 0.05,差不多需要40 000次迭代
#runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)

#对比不同的梯度下降方法
#Stochastic descent
#runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.001)

#有点爆炸。。。很不稳定,再来试试把学习率调小一些
#runExpe(orig_data, theta, 1, STOP_ITER, thresh=15000, alpha=0.000002)
#速度快，但稳定性差，需要很小的学习率

#Mini-batch descent
#runExpe(orig_data, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)
#浮动仍然比较大，我们来尝试下对数据进行标准化 将数据按其属性(按列进行)减去其均值，然后除以其方差。最后得到的结果是，对每个属性/每列来说所有数据都聚集在0附近，方差值为1
from sklearn import preprocessing as pp

scaled_data = orig_data.copy()
scaled_data[:, 1:3] = pp.scale(orig_data[:, 1:3])

#runExpe(scaled_data, theta, n, STOP_ITER, thresh=5000, alpha=0.001)
#它好多了！原始数据，只能达到达到0.61，而我们得到了0.38个在这里！ 所以对数据做预处理是非常重要的
#runExpe(scaled_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)

#更多的迭代次数会使得损失下降的更多！
theta = runExpe(scaled_data, theta, 1, STOP_GRAD, thresh=0.002/5, alpha=0.001)
#随机梯度下降更快，但是我们需要迭代的次数也需要更多，所以还是用batch的比较合适！！！
#runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)

#精度
#设定阈值
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]

scaled_X = scaled_data[:, :3]
y = scaled_data[:, 3]
predictions = predict(scaled_X, theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))