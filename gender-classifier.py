
# coding: utf-8

# In[153]:


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


# In[154]:


dataset = pd.read_csv("gender-classifier-DFE-791531.csv", encoding='latin1')
dataset.head()


# In[155]:


numnull = dataset.isnull().mean()


# In[156]:


list_cols = numnull[numnull < 0.30].index.values.tolist()
dataset = dataset[list_cols]


# In[157]:


print(dataset.shape)
print(dataset.dropna().shape)


# In[158]:


dataset = dataset.dropna()


# In[159]:


def normalize_text(s):
    # just in case
    s = str(s)
    s = s.lower()
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    
    return s

#dataset['text_norm'] = [normalize_text(s) for s in dataset['text']]
#dataset['description_norm'] = [normalize_text(s) for s in dataset['description']]

dataset['text_norm'] = dataset.text.apply(normalize_text)
dataset['description_norm'] = dataset.description.apply(normalize_text)


# In[160]:


# how many observations are gold standard?
gold_values = defaultdict(int)
for val in dataset._golden:
    gold_values[val] += 1
print(gold_values)

# what does the confidence look like?
print(np.any(np.isnan(dataset['gender:confidence'])))
# we've got at least one NaN, so let's remove
gender_confidence = dataset['gender:confidence'][np.where(np.invert(np.isnan(dataset['gender:confidence'])))[0]]
print(len(gender_confidence))
gender_nonones = gender_confidence[np.where(gender_confidence < 1)[0]]
print(len(gender_nonones))


# In[161]:


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


# In[162]:


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


# In[189]:


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0 # SVM regularization parameter

models = [svm.SVC(kernel='linear', C=C), svm.LinearSVC(C=C),
svm.SVC(kernel='rbf', gamma = 0.7, C = C),
svm.SVC(kernel='poly', degree = 3, C = C)]


# In[190]:


# pull the data into vectors
vectorizer = TfidfVectorizer(max_features = 300)
x1 = pd.DataFrame(vectorizer.fit_transform(dataset_confident['text']).todense()).reset_index()
x2 = pd.get_dummies(dataset_confident['sidebar_color']).reset_index()
x3 = dataset_confident['tweet_count']
x = pd.concat([x1, x2], axis = 1)


# In[ ]:




encoder = LabelEncoder()
y = encoder.fit_transform(dataset_confident['gender'])


models = [clf.fit(x, y) for clf in models] 

# title for the plots
titles = ('SVC with linear kernel',
'LinearSVC (linear kernel)',
'SVC with RBF kernel',
'SVC with polynomial (degree 3) kernel')



# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)

    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('the xes')
    ax.set_ylabel('gender')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

