import numpy as np
import matplotlib.pyplot as plt


def createPlot(figure_index, name, x, y, y_predict=None, theta=None, xLabel='x', yLabel='y', textLocationX=0.4,
               textLocationY=3.5, xmin = 0,xmax=1,ymin=3,ymax=5):
    fig = plt.figure(figure_index)
    fig.suptitle(name)
    plt.plot(x, y, 'ro', label='original point', markersize=5)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.axis([xmin,xmax,ymin,ymax])
    if (y_predict is not None):
        plt.plot(x, y_predict, label='best fit line', markersize=5)
        if (len(theta) == 2):
            plt.text(textLocationX, textLocationY, yLabel + ' = ' + str(theta[0][0]) + '+' + str(theta[1][0]) + xLabel)
        elif (len(theta == 3)):
            plt.text(0.15, 3.05, yLabel + ' = ' + str(theta[0][0]) + '+' + str(theta[1][0]) + xLabel + '+' + str(
                theta[2][0]) + xLabel + '^2')
    plt.legend()


def loadDataSet(fname):
    data = np.loadtxt(fname)
    xVal = data[:, :2]
    yVal = data[:, 2:3]
    return xVal, yVal


def standRegressNormalEquation(xVal, yVal):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(xVal), xVal)), np.transpose(xVal)), yVal)
    return theta


def standRegressGD(xVal, yVal):
    numOfIteration = 5000
    alpha = 0.0005
    theta = np.ones(2).reshape(2, 1)
    i = 0
    while (i < numOfIteration):
        gradient = np.dot(np.transpose(xVal), (np.dot(xVal, theta) - yVal))
        theta = theta - alpha * gradient
        i = i + 1
    return theta


def polyRegressNormalEquation(xVal, yVal):
    x1 = xVal[:, 1:]
    fi = np.insert(xVal, 2, np.transpose(x1 ** 2), axis=1)
    theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(fi), fi)), np.transpose(fi)), yVal)
    return theta


def polyRegressGD(xVal, yVal):
    x1 = xVal[:, 1:]
    fi = np.insert(xVal, 2, np.transpose(x1 ** 2), axis=1)
    numOfIteration = 8000
    alpha = 0.005
    theta = np.ones(3).reshape(3, 1)
    i = 0
    while (i < numOfIteration):
        gradient = np.dot(np.transpose(fi), (np.dot(fi, theta) - yVal))
        theta = theta - alpha * gradient
        i = i + 1
    return theta


def standRegress(xVal, yVal, mode='Normal Equation'):
    if (mode == 'Normal Equation'):
        theta = standRegressNormalEquation(xVal, yVal)
    elif (mode == 'GD'):
        theta = standRegressGD(xVal, yVal)
    return theta


def polyRegress(xVal, yVal, mode='Normal Equation'):
    if (mode == 'Normal Equation'):
        theta = polyRegressNormalEquation(xVal, yVal)
    elif (mode == 'GD'):
        theta = polyRegressGD(xVal, yVal)
    return theta


if __name__ == '__main__':
    (x, y) = loadDataSet("Q2data.txt")
    x1 = x[:, 1:]
    createPlot(1, 'p1_1', x1, y)
    theta = standRegress(x, y, 'Normal Equation')
    createPlot(2, 'p1_2_' + 'Normal Equation', x1, y, np.dot(x, theta), theta)
    theta = standRegress(x, y, 'GD')
    createPlot(3, 'p1_2_' + 'GD', x1, y, np.dot(x, theta), theta)
    theta = polyRegress(x, y, 'Normal Equation')
    x1 = x[:, 1:]
    fi = np.insert(x, 2, np.transpose(x1 ** 2), axis=1)
    createPlot(4, 'p1_3_' + 'Normal Equation', x1, y, np.dot(fi, theta), theta)
    theta = polyRegress(x, y, 'GD')
    createPlot(5, 'p1_3_' + 'GD', x1, y, np.dot(fi, theta), theta)
    plt.show()
