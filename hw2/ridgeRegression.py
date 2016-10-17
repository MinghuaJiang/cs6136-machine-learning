import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import linearRegression

def create3DPlot(figure_index, name, x, y, y_predict=None):
    fig = plt.figure(figure_index)
    fig.suptitle(name)
    ax = fig.add_subplot(111, projection='3d')
    x1 = x[:, 1:2]
    x2 = x[:, 2:]
    ax.scatter(x1, x2, y)
    if(y_predict is not None):
        x1,x2 = np.meshgrid(x1, x2)
        ax.plot_surface(x1,x2,np.dot(x, beta))

def createPlot(figure_index, name, x,y):
    fig = plt.figure(figure_index)
    fig.suptitle(name)
    plt.plot(x, y, 'ro')
    plt.xlabel('Lambda')
    plt.ylabel('MSE')

def loadDataSet(fname):
    data = np.loadtxt(fname)
    xVal = data[:,:3]
    yVal = data[:,3:]
    return xVal,yVal

def ridgeRegress(xVal, yVal, lambd = 0):
    x = xVal[:,1:3]
    identity = np.identity(len(np.transpose(x)))
    beta0 = np.mean(yVal)
    betaWithoutBeta0 = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x) + np.dot(lambd,identity)), np.transpose(x)), yVal - beta0)
    beta0Array = np.array(beta0)
    beta = np.insert(betaWithoutBeta0,0,beta0Array, axis=0)
    return beta

def cv(xVal, yVal):
    np.random.seed(37)
    np.random.shuffle(xVal)
    np.random.seed(37)
    np.random.shuffle(yVal)
    splittedXArray = np.split(xVal,10)
    splittedYArray = np.split(yVal,10)
    lamdaMap = dict()
    dataRows = len(yVal)
    kFolder = 10
    trainDataRows = dataRows * (kFolder - 1) / kFolder
    testDataRows = dataRows - trainDataRows

    for lamdaIndex in range(1,50):
        lamda = 0.02 * lamdaIndex
        averageMSE = 0
        for i in range(0, kFolder):
            testXData = splittedXArray[i]
            trainXData = np.zeros((trainDataRows, len(xVal[0])))
            testYData = splittedYArray[i]
            trainYData = np.zeros((trainDataRows, len(yVal[0])))

            for j in range(0, i):
                for k in range(0, testDataRows):
                    trainXData[j * testDataRows + k] = splittedXArray[j][k]
                    trainYData[j * testDataRows + k] = splittedYArray[j][k]
            for j in range(i + 1, kFolder):
                for k in range(0, testDataRows):
                    trainXData[(j - 1) * testDataRows + k] = splittedXArray[j][k]
                    trainYData[(j - 1) * testDataRows + k] = splittedYArray[j][k]

            beta = ridgeRegress(trainXData, trainYData, lamda)
            y_predict = np.dot(testXData,beta)
            mse = np.dot(np.transpose(testYData - y_predict),testYData - y_predict) / testDataRows
            averageMSE += mse[0][0]

        averageMSE = averageMSE / kFolder
        lamdaMap[averageMSE] = lamda

    minMSE = min(lamdaMap.keys())
    lambdaBest = lamdaMap[minMSE]
    createPlot(3,'1.4.2 Lambda VS MSE', lamdaMap.values(),lamdaMap.keys())
    return lambdaBest

if __name__ == '__main__':
    (x,y) = loadDataSet("RRdata.txt")
    create3DPlot(1, '1.4.0', x, y)
    beta = ridgeRegress(x,y)
    print(beta)
    create3DPlot(2, '1.4.1 Ridge Regression with Lambda 0', x, y, np.dot(x, beta))
    lambdaBest = cv(x,y)
    beta = ridgeRegress(x, y, lambdaBest)
    create3DPlot(4, '1.4.2 Ridge Regression with Best Lambda ' + str(lambdaBest), x, y, np.dot(x, beta))
    (x, y) = linearRegression.loadDataSet("RRdata.txt")
    theta = linearRegression.standRegress(x, y, 'Normal Equation')
    x1 = x[:, 1:]
    linearRegression.createPlot(5, '1.5 X2 VS X1', x1, y, np.dot(x, theta), theta, 'x1', 'x2', -1.5, -4, -4, 4, -7, 9)
    plt.show()