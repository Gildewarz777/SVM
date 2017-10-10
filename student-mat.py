
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, svm
import re
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, auc

dataset = pd.read_csv("./dataset/student-mat.csv", encoding='latin1')
list(dataset)
dataset.head()

g = dataset.isnull().mean()

print(dataset.shape)

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in
    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

C = 1.0

u = np.random.randint(0, 3, dataset.shape[0])
dataset_train = dataset[u != 0]
dataset_test = dataset[u == 0]


x0 = dataset_train.absences
x1 = dataset_train.freetime
x2 = dataset_train.age
x3 = dataset_train.Medu
x4 = dataset_train.Fedu
x5 = dataset_train.goout
x_train = pd.concat([x0, x1, x2, x3, x4, x5], axis = 1)

x0 = dataset_test.absences
x1 = dataset_test.freetime
x2 = dataset_test.age
x3 = dataset_test.Medu
x4 = dataset_test.Fedu
x5 = dataset_test.goout
x_test = pd.concat([x0, x1, x2, x3, x4, x5], axis = 1)

print(x_train.shape)
print(x_test.shape)


y_train = np.ravel(pd.get_dummies(dataset_train.schoolsup, drop_first=True))
y_test = np.ravel(pd.get_dummies(dataset_test.schoolsup, drop_first=True))

print("y_train shape ", y_train.shape)
print("y_test shape ", y_test.shape)

for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    print("training new model")
    clf = svm.SVC(kernel=kernel, gamma=0.01, C=100)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("Score", score)

"""models = (svm.SVC(kernel='linear', C=C), svm.LinearSVC(C=C), svm.SVC(kernel='rbf', gamma=0.7, C=C), svm.SVC(kernel='poly', degree=3, C=C))

models = (clf.fit(x_train, y_train) for clf in models)
score = clf.score(x_test, y_test)
print(list(score))"""

