import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import sklearn.utils as utils
import sklearn.linear_model as lin
import bonnerlib2
import pickle

# Q1

def genData (mu0, mu1, N):

    data0 = mu0 + rnd.randn(N, 2)
    data1 = mu1 + rnd.randn(N, 2)
    X = np.concatenate([data0, data1])
    t0 = np.zeros([N], dtype=np.int32)
    t1 = np.ones([N], dtype=np.int32)
    t = np.concatenate([t0, t1])

    return utils.shuffle(X, t)


N = 10000
mu0 = np.array([1.0, -1.0])/2
mu1 = np.array([-1.0, 1.0])/2
X,t = genData(mu0, mu1, N)

colors = np.array(['r','b'])
plt.figure()
plt.scatter(X[:,0], X[:, 1], color=colors[t], s=10)
plt.suptitle('Figure 1: scatter plot of data')



def graphScatter(mu0,mu1,N):
    X,t = genData(mu0,mu1,N)
    colors = np.array(['r','b'])
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], color = colors[t], s=1)
    


# Q2

def logregDemo(N,betaList):
    I = len(betaList)
    M = np.int(np.ceil(np.sqrt(I)))
    mu0 = np.array([2.0, -2.0])
    mu1 = np.array([-2.0, 2.0])
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig1.suptitle('Figure 2: contour plots of logistic decision functions')
    fig2.suptitle('Figure 3: surface plots of logistic decision functions')

    for i in range(I):
        beta = betaList[i]
        X,t = genData(beta * mu0, beta * mu1, N)

        clf = lin.LogisticRegression()
        clf.fit(X, t)

        acc = clf.score(X, t)
        print ('For beta={}, the accuracy is {}'.format(beta, acc))

        ax = fig1.add_subplot(M, M, i + 1)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        colors = np.array(['r', 'b'])
        ax.scatter(X[:, 0], X[:, 1], color= colors[t], s = 0.1)
        bonnerlib2.dfContour(clf, ax)

        ax = fig2.add_subplot(M, M, i + 1, projection = '3d')
        ax.set_xlim(-9, 6)
        ax.set_ylim(-6, 9)
        bonnerlib2.df3D(clf, ax)

print('\n')
print('Question 2(b).')
print('---------------')
logregDemo(10000, [0.1, 0.2, 0.5, 1.0])



#Q3

with open ('mnist.pickle', 'rb') as f:
    data = pickle.load(f)

Xtrain = data['training']
Xtest = data['testing']

# 3(a)

def displaySample(N, data):
    M = int(np.ceil(np.sqrt(N)))
    m = int(np.sqrt(np.size(data[0])))
    sample = utils.resample(data, n_samples = N, replace= False)

    for i in range(0,N):
        x = sample[i]
        y = np.reshape(x, (m, m))
        plt.subplot(M, M, i + 1)
        plt.axis('off')
        plt.imshow(y, cmap = 'Greys', interpolation = 'nearest')

plt.figure()
plt.suptitle('Figure 4: random MNIST images of the digit 5')
displaySample(14, Xtrain[5])


# 3(b)

flatTrain = np.vstack(Xtrain)
plt.figure()
plt.suptitle('Figure 5: random sample of MNIST training images')
displaySample(23, flatTrain)


# 3(c)


def flatten(data):
    K = len(data)
    X = np.vstack(data)
    N = np.shape(X)[0]
    t = np.zeros(N, dtype = 'int')
    m1 = 0
    m2 = 0

    for i in range(0, K):
        m = np.shape(data[i])[0]
        m2 += m
        t[m1:m2] = i
        m1 += m

    return X, t


# 3(d)

print('\n')
print('Question 3(d).')
print('---------------')
X, t = flatten(Xtrain)
clf = lin.LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
clf.fit(X, t)
acc = clf.score(X, t)
print('Training accuracy: {}'.format(acc))
X,t = flatten(Xtest)
acc = clf.score(X, t)
print('Test accuracy: {}'.format(acc))


# 3(e)

y = clf.predict(X)
idx = (y != t)
Xerr = X[idx]
fig = plt.figure()
fig.suptitle('Figure 6: some misclassified images')
displaySample(36, Xerr)


# 3(f)


P = clf.predict_proba(X)
Pmax = np.max(P, axis = 1)
idx = np.argsort(Pmax)
Xsorted = X[idx]
fig = plt.figure()
fig.suptitle('Figure 7: images with the least confident predictions')
displaySample(16, Xsorted[:16])



# 4(d)

print('\n')
print('Question 4(d).')
print('---------------')

X,t = flatten((Xtrain[2], Xtrain[3]))
clf = lin.LogisticRegression(solver = 'lbfgs')
clf.fit(X, t)
X,t = flatten((Xtrain[2], Xtrain[3]))
acc = clf.score(X, t)
print('Test accuracy: {}'.format(acc))










    





