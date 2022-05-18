import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import statistics
import numpy as np
import matplotlib.pyplot as plt
from heapq import nsmallest
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc


######################################################################################################################
# Preliminary Analysis        #
###############################
wsdf_pre = pd.read_csv("wholesale.csv")
wsdf_pre.loc[:, 'Channel'] = None
wsdf_pre.loc[:, 'Region'] = None
wsdf_pre = wsdf_pre.drop(['Channel', 'Region'], axis=1)

#print(wsdf.isnull().any()) # No missing values found

transformeddf = wsdf_pre
lambda_list = []
for column in transformeddf:
    # Boxcox transform training data & save lambda value
    column_data = transformeddf[column]
    fitted_data, fitted_lambda = stats.boxcox(column_data)
    fitted_data = pd.DataFrame(fitted_data, columns=[column])
    # Standardize data around zero
    scaler = StandardScaler()
    X = scaler.fit_transform(fitted_data)
    scaled_data = pd.DataFrame(X, columns=[column])
    transformeddf.update(scaled_data)
    lambda_list.append(fitted_lambda)
wsdf = transformeddf
print(wsdf.head())

pca = PCA(2)
'''
######################################################################################################################
# Question 1a         #    Elbow Plot
#######################
distortions = []
K = range(1,16)
for k in K:
    kmeanModel = KMeans(n_clusters=k, max_iter=75, n_init=50)
    kmeanModel.fit(wsdf)
    distortions.append(kmeanModel.inertia_)
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# No obvious elbow exists in the data

######################################################################################################################
# Question 1b         #    K Means Clustering
#######################
kmeanModel = KMeans(n_clusters=3, max_iter=75, n_init=50, random_state=577)
kmeanModel.fit(wsdf)
labels = list(kmeanModel.labels_)
print("Clients in each cluster\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}".format(labels.count(0), labels.count(1),
      labels.count(2)))
centers = list(kmeanModel.cluster_centers_)
for i in centers:
    k = 0
    for j in i:
        i[k] = round(i[k], 2)
        k+=1
print(centers)
print("Cluster centers:\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}".format(centers[0], centers[1],
      centers[2]))

pca_data = pd.DataFrame(pca.fit_transform(wsdf),columns=['PC1','PC2'])
pca_data['cluster'] = pd.Categorical(kmeanModel.labels_)
sns.scatterplot(x="PC1",y="PC2",hue="cluster",data=pca_data)
plt.show()

# Cluster 1 occurs when the Fresh, Frozen, and Delicassen categories are all high.
# Cluster 3 occurs when the remaining categories, Milk, Grocery, and Detergents are high.
# Cluster 2 occurs when Milk, Grocery, and Detergents are very low

######################################################################################################################
# Question 1c         #    K Means Clustering
#######################
kmeanModel = KMeans(n_clusters=4, max_iter=75, n_init=50, random_state=577)
kmeanModel.fit(wsdf)
labels = list(kmeanModel.labels_)
print("Clients in each cluster\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}\nCluster 4: {}".format(
    labels.count(0), labels.count(1),labels.count(2), labels.count(3)))
centers = list(kmeanModel.cluster_centers_)
for i in centers:
    k = 0
    for j in i:
        i[k] = round(i[k], 2)
        k+=1
print(centers)
print("Cluster centers:\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}\nCluster 4: {}".format(centers[0], centers[1],
      centers[2], centers[3]))

pca_data = pd.DataFrame(pca.fit_transform(wsdf),columns=['PC1','PC2'])
pca_data['cluster'] = pd.Categorical(kmeanModel.labels_)
sns.scatterplot(x="PC1",y="PC2",hue="cluster",data=pca_data)
plt.show()

# Using PCA analysis to visualize the two clustering models using 3 and 4 clusters, it appears that when a fourth
# cluster is added, it results from one of the other clusters being split in half. There is no apparent distinction
# between the two clusters. Therefore, I believe adding the fourth cluster does not significantly help the model.

######################################################################################################################
# Question 2a         #    K Means Clustering
#######################
plt.figure(figsize=(10, 7))
plt.title("Wholesale Dendograms")
dend = shc.dendrogram(shc.linkage(wsdf, method='complete'))
plt.show()

######################################################################################################################
# Question 2b         #    K Means Clustering
#######################
plt.figure(figsize=(10, 7))
plt.title("Wholesale Dendograms")
dend = shc.dendrogram(shc.linkage(wsdf, method='ward'))
plt.axhline(linestyle='--', y=11)
plt.show()

# By cutting at this height, we would have six clusters
'''
######################################################################################################################
# Question 2c         #
#######################
acmodel = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
acmodel.fit_predict(wsdf)
labels = list(acmodel.labels_)
print("Clients in each cluster\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}\nCluster 4: {}".format(
    labels.count(0), labels.count(1),labels.count(2), labels.count(3)))

adf = pd.DataFrame(columns=['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'])
bdf = pd.DataFrame(columns=['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'])
cdf = pd.DataFrame(columns=['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'])
ddf = pd.DataFrame(columns=['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'])
i = 0
for client in labels:
    if client == 0:
        adf = adf.append(wsdf.loc[i])
    if client == 1:
        bdf = bdf.append(wsdf.loc[i])
    if client == 2:
        cdf = cdf.append(wsdf.loc[i])
    if client == 3:
        ddf = ddf.append(wsdf.loc[i])
    i += 1
centroids = []
cluster_list = [adf, bdf, cdf, ddf]
for cluster in cluster_list:
    fresh_center = statistics.mean(list(cluster.loc[:,"Fresh"]))
    milk_center = statistics.mean(list(cluster.loc[:, "Milk"]))
    grocery_center = statistics.mean(list(cluster.loc[:, "Grocery"]))
    frozen_center = statistics.mean(list(cluster.loc[:, "Frozen"]))
    detergents_center = statistics.mean(list(cluster.loc[:, "Detergents_Paper"]))
    delicassen_center = statistics.mean(list(cluster.loc[:, "Delicassen"]))
    current_list = [fresh_center, milk_center, grocery_center, frozen_center, detergents_center, delicassen_center]
    centroids.append(current_list)

print("Cluster centers:\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}\nCluster 4: {}".format(centroids[0], centroids[1],
      centroids[2], centroids[3]))

# These appear to be different clusters than the KMeans method, with centroids in different locations.

pca_data = pd.DataFrame(pca.fit_transform(wsdf),columns=['PC1','PC2'])
pca_data['cluster'] = pd.Categorical(acmodel.labels_)
sns.scatterplot(x="PC1",y="PC2",hue="cluster",data=pca_data)
plt.show()

######################################################################################################################
# Question 2d         #
#######################
acmodel = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
acmodel.fit_predict(wsdf)
pca_data = pd.DataFrame(pca.fit_transform(wsdf),columns=['PC1','PC2'])
pca_data['cluster'] = pd.Categorical(acmodel.labels_)
sns.scatterplot(x="PC1",y="PC2",hue="cluster",data=pca_data)
plt.show()

# From the chart using 2D principle components analysis, it appears that the fifth cluster was created from the center
# area between the previous four. As a result, there is significant overlap between the fifth cluster and the other
# clusters. From this analysis, it does not appear that the additional cluster improves the analysis in finding a
# distinct grouping of data.