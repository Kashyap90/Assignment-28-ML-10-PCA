
# coding: utf-8

# In[1]:


#Problem Statement

#In this assignment students have to transform iris data into 3 dimensions and plot a 3d chart with transformed dimensions 
#and color each data point with specific class.


# In[2]:


# Import libraries into working environment:


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
import seaborn as sns
from sklearn.decomposition import PCA


# In[4]:


# Load iris data set:


# In[5]:


iris = datasets.load_iris()
X = iris.data
y = iris.target
print("Number of samples:")

print(X.shape[0])
print('------------------------------------------------------------------------------------')
print('Number of features :')
print(X.shape[1])
print('------------------------------------------------------------------------------------')
print("Feature names:")
print('------------------------------------------------------------------------------------')
print(iris.feature_names)


# In[6]:


# Feature scaling prior to applying PCA:


# In[7]:


# Feature Scaling:

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
print('Shape of scaled data points:')
print('----------------------------------------------------------------------')
print(X_scaled.shape)
print('----------------------------------------------------------------------')
print('First 5 rows of scaled data points :')
print('----------------------------------------------------------------------')
print(X_scaled[:5, :])


# In[8]:


# looking at the explained variance as a function of the components:


# In[9]:


sns.set()
pca = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# In[10]:


# Here we see that we'd need about 3 components to retain 100% of the variance. 
#Looking at this plot for a high-dimensional dataset can help us understand the level of redundancy present in multiple 
#observations.


# In[11]:


# PCA using Eigen-decomposition: 5-step process:


# In[12]:


# 1. Normalize columns of A so that each feature has zero mean:

A0 = iris.data
mu = np.mean(A0,axis=0)
A = A0 - mu
print("Does A have zero mean across rows?")
print(np.mean(A,axis=0))
print('--------------------------------------------------------------------------')
print('Mean value : ')
print('--------------------------------------------------------------------------')
print(mu)
print('Standardized Feature value first 5 rows: ')
print('--------------------------------------------------------------------------')
print(A[:5,:])

# 2. Compute sample covariance matrix Sigma = {A^TA}/{(m-1)}
#covariance matrix can also be computed using np.cov(A.T):

m,n = A.shape
Sigma = (A.T @ A)/(m-1)
print("--------------------------------------------------------------------------")
print("Sigma:")
print(Sigma)

# 3. Perform eigen-decomposition of Sigma using `np.linalg.eig(Sigma):

W,V = np.linalg.eig(Sigma)
print("---------------------------------------------------------------------------")
print("Eigen values:")
print(W)
print("---------------------------------------------------------------------------")
print("Eigen vectors:")
print(V)

# 4. Compress by ordering 3 eigen vectors according to largest eigen values and compute AX_k:

print("----------------------------------------------------------------------------")
print("Compressed - 4D to 3D:")
print("----------------------------------------------------------------------------")
print('First 3 eigen vectors :')
print(V[:,:3] )
print("----------------------------------------------------------------------------")
Acomp = A @ V[:,:3] 
print('First first five rows of transformed features :')
print("----------------------------------------------------------------------------")
print(Acomp[:5,:]) 


# 5. Reconstruct from compressed version by computing $A V_k V_k^T$:

print("----------------------------------------------------------------------------")
print("Reconstructed version - 3D to 4D:")
print("----------------------------------------------------------------------------")
Arec = A @ V[:,:3] @ V[:,:3].T # first 3 evectors
print(Arec[:5,:]+mu) # first 5 obs, adding mu to compare to original


# In[13]:


# Original iris feature values:


# In[14]:


iris.data[:5, :]


# In[15]:


# 3D Visualization:


# In[16]:


np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)
y= iris.target
plt.cla()
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(Acomp[y == label, 0].mean(),
              Acomp[y == label, 1].mean() + 1.5,
              Acomp[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(Acomp[:, 0], Acomp[:, 1], Acomp[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[17]:


# Applying PCA for number of compents = 3 using sklearn:


# In[18]:


pca = PCA(n_components=3)
pca.fit(X_scaled)
print('explained variance :')
print('--------------------------------------------------------------------')
print(pca.explained_variance_)
print('--------------------------------------------------------------------')
print('PCA Components : ')
print('--------------------------------------------------------------------')
print(pca.components_)
print('--------------------------------------------------------------------')
X_transformed = pca.transform(X)
print('Transformed Feature values first five rows :')
print('--------------------------------------------------------------------')
print(X_transformed[:5,:])
print('--------------------------------------------------------------------')
print('Transformed Feature shape :')
print('--------------------------------------------------------------------')
print(X_transformed.shape)
print('--------------------------------------------------------------------')
print('Original Feature shape :')
print('--------------------------------------------------------------------')
print(X.shape)
print('--------------------------------------------------------------------')
print('Retransformed  Feature  :')
print('--------------------------------------------------------------------')

X_retransformed = pca.inverse_transform(X_transformed)

print('Retransformed Feature values first five rows :')
print('--------------------------------------------------------------------')
print(X_retransformed[:5,:])


# In[19]:


# Note:

#Transformed from 4D to 3D using PCA


# In[20]:


print('First Principal Component PC1: ', pca.components_[0])
print('\nSecond Principal Component PC2: ', pca.components_[1])
print('\nThird Principal Component PC3 :', pca.components_[2])


# In[21]:


# Note:

#Transforming from 3D to 4D


# In[22]:


# 3D Visualization:


# In[23]:


np.random.seed(5)
centers = [[1, 1], [-1, -1], [1, -1]]
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)
y= iris.target
plt.cla()
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X_transformed[y == label, 0].mean(),
              X_transformed[y == label, 1].mean() + 1.5,
              X_transformed[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    
# Reorder the labels to have colors matching the cluster results

y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

