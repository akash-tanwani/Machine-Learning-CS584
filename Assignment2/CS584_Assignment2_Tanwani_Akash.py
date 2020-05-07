# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:23:56 2020

@author: Akash1313
"""

#QUESTION 1:
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori

# load data file
dataset = pd.read_csv(r"C:\Users\Akash1313\Desktop\CS584_ML\Assignment2\Groceries.csv", delimiter=',')
print(len(dataset))

# Que 1 part(a)
new_dataset = dataset.groupby(['Customer'])['Item'].count()
new_dataset = new_dataset.sort_values()
print("Dataset is: \n ",dataset)
print("\nDistinct items :\n",new_dataset)

Q1,Q2,Q3= np.percentile(new_dataset, [25 ,50 ,75])
print("the 25th percentiles of the histogram is:",Q1)
print("Median of the histogram is",Q2)
print("the 75th percentiles of the histogram is:",Q3)

#histogram
plt.hist(new_dataset,color="gray")
plt.title('Histogram for the number of unique items')
plt.xlabel('Item Label')
plt.ylabel('Frequency count')
plt.axvline(Q1, color='red', linewidth=1, alpha=1)
plt.axvline(Q2, color='red', linewidth=1, alpha=1)
plt.axvline(Q3, color='red', linewidth=1, alpha=1)
plt.grid(True)
plt.show()

# Que 1 part(b)
# Convert the data to the Item List format
listItem = dataset.groupby(['Customer'])['Item'].apply(list).values.tolist()
# Convert the Item List format to the Item Indicator format
te = TransactionEncoder()
te_ary = te.fit(listItem).transform(listItem)
itemIndicator = pd.DataFrame(te_ary, columns=te.columns_)
# Find the frequent itemsets
min_supp = 75 / len(new_dataset)
print("\nMinimum support is:",min_supp)
frequent_itemset = apriori(itemIndicator, min_support=min_supp, use_colnames=True)
k_largest_itemset = len(frequent_itemset['itemsets'][len(frequent_itemset) - 1])
print("frequent item sets : \n",frequent_itemset['itemsets'])
print("\ntotal number of item sets found = ",frequent_itemset.shape[0])
print("\nthe largest value of k =",k_largest_itemset)

# Que 1 part (c)
# Discover the association rules
#confidence=1/100
assoc_rules = association_rules(frequent_itemset, metric="confidence", min_threshold=0.01)
print("\nTotal number of association rules found = ",assoc_rules.shape[0])

# Que 1 part (d)
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], c=assoc_rules['lift'], s=assoc_rules['lift'])
plt.ylabel("Support")
plt.xlabel("Confidence")
cbar = plt.colorbar()
cbar.set_label('lift', labelpad=+1)
plt.grid(True)
plt.show()

#Que 1 part(e)
# Discover the association rules
assoc_rules = association_rules(frequent_itemset, metric="confidence", min_threshold=0.6)
print("association rules are: \n",assoc_rules.to_string())

############################################################
#QUESTION 2:
# Load the libraries
import pandas as pd

# load data file
dataset = pd.read_csv(r"C:\Users\Akash1313\Desktop\CS584_ML\Assignment2\cars.csv", delimiter=',')
print(dataset)

#Que 2 part(a)
print('\nThe frequencies of the categorical feature Type is ')
print(dataset['Type'].value_counts())

#Que 2 part(b)
print('\nThe frequencies of the categorical feature DriveTrain is ')
print(dataset['DriveTrain'].value_counts())

#Que 2 part(c)
val = dataset['Origin'].value_counts(dropna = False).to_dict()
dist = 1/(val['Asia'])+1/(val['Europe'])
print('\ndistance between Origin = ‘Asia’ and Origin = ‘Europe’is ',dist)

#Question 2 part(d)
val1 = dataset['Cylinders'].fillna(0.0).value_counts(dropna = False).to_dict()
print(val1)
dist1 = 1/(val1[5.0])+1/(val1[0.0])
print('\nThe the distance between Cylinders = 5 and Cylinders = Missing is ',dist1)

#Que 2 part(e)
from kmodes.kmodes import KModes
cardata = dataset[['Type','Origin','DriveTrain','Cylinders']]
cardata = cardata.fillna(-1)
km = KModes(n_clusters=3,init='Huang')
clusters = km.fit_predict(cardata)
print('\nThe number of observations in each clusters are: ')
print("\t0:",list(clusters).count(0))
print("\t1:",list(clusters).count(1))
print("\t2:",list(clusters).count(2))
print('The centroids for the 3 clusters are:')
print(km.cluster_centroids_)

#Que 2 (f)
cardata['Cluster_number']=pd.Series(clusters)
print("\n",pd.crosstab(index = cardata["Origin"], columns = cardata["Cluster_number"]))

############################################################
#QUESTION 3:
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.neighbors as neighbors
import numpy as np
import math
from numpy import linalg
# load data file
dataset = pd.read_csv(r"C:\Users\Akash1313\Desktop\CS584_ML\Assignment2\FourCircle.csv", delimiter=',')
dataset = dataset.dropna()
print(dataset)

# Que 3 part(a)
plt.scatter(dataset['x'], dataset['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Que 3 part(b)
# finding clusters in plot
num_clusters = 4
trainData = dataset[['x', 'y']]
k_means_model = cluster.KMeans(n_clusters=num_clusters, random_state=60616)
k_means=k_means_model.fit(trainData)
print("Centroids of the Cluster are= \n", k_means.cluster_centers_)

# printing cluster data
dataset['k_mean_cluster'] = k_means.labels_
for i in range(num_clusters):
    print("\ncluster label = ", i)
    print(dataset.loc[dataset['k_mean_cluster'] == i])
#plot
plt.scatter(dataset['x'], dataset['y'], c=dataset['k_mean_cluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

#Que 3 part(c and d)
kNNSpec = neighbors.NearestNeighbors(n_neighbors = 15,algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
dis, ind = nbrs.kneighbors(trainData)

# Retrieve the distances among the observations
distObject = neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency matrix
nObs = dataset.shape[0]
Adjacency = np.zeros((nObs, nObs))
for i in range(nObs):
    for j in ind[i]:
        Adjacency[i,j] = math.exp(- (distances[i][j])**2 )

# Make the Adjacency matrix symmetric
Adjacency = 0.5 * (Adjacency + Adjacency.transpose())
print("\nAdjacency Matrix is:",Adjacency)
# Create the Degree matrix
Degree = np.zeros((nObs, nObs))
for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
print("\nDegree Matrix is:",Degree)
    
# Create the Laplacian matrix        
Lmatrix = Degree - Adjacency
print("\nLaplacian Matrix is:",Lmatrix)
# Obtain the eigenvalues and the eigenvectors of the Laplacian matrix
evals, evecs = linalg.eigh(Lmatrix)

# Series plot of the eigenvalues to determine the number of neighbors
sequence = np.arange(1,16,1) 
plt.plot(sequence, evals[0:15,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.grid("both")
plt.xticks(sequence)
plt.show()

for i in range(4):
    print("{:e}".format(evals[i]))
    
Z = evecs[:,[0,1]]
plt.scatter(1e10*Z[:,0], Z[:,1])
plt.xlabel('First Eigenvector')
plt.ylabel('Second Eigenvector')
plt.grid("both")
plt.show()
    
# Que 3 part(e)
Z = evecs[:,[0,3]]
kmeans_spectral = cluster.KMeans(n_clusters=4, random_state=60616).fit(Z)
dataset['ring'] = kmeans_spectral.labels_
plt.scatter(dataset['x'], dataset['y'], c=dataset['ring'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()