# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import math
import seaborn as sns
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

#Question 1
dataset=pd.read_csv(r"C:\Users\Akash1313\Desktop\CS584_ML\Assignment1\NormalSample.csv")
print(dataset.head())

#Question 1 part a
Q1,Q3= np.percentile(dataset.x, [25 ,75])
IQR=Q3-Q1
N=len(dataset)
h=2*IQR*(N**(-1/3))
print("The recommended bin-width for the histogram of x by using Izenman (1991) method will be given by: ",h,"=",round(h, 2))

#Question 1 part b
minimum_x,maximum_x=min(dataset.x),max(dataset.x)
print("Minimum value of field 'x' is: ",minimum_x)
print("Maximum value of field 'x' is: ",maximum_x)

#Question 1 part c
a=math.floor(minimum_x)
b=math.ceil(maximum_x)
print("The largest integer less than the minimum value of the field x will be: ",a)
print("The smallest integer greater than the maximum value of the field x will be: ",b)

#Estimation of density:
def estimate_density(h,a,b):
    mid_points=[]
    point= a+h/2
    while point<b:
        mid_points.append(point)
        point+=h
    
    density=[]
    for i in mid_points:
        modified=(dataset.x-i)/h
        temp_sum=0
        for j in modified:
            if j>= -0.5 and j<=0.5:
                temp_sum+=1
        density.append(temp_sum/(len(dataset.x)*h))
    
    plt.step(mid_points,density,color='r', where = 'mid')    
    plt.grid(True)
    plt.show() 
    data = pd.DataFrame(list(zip(mid_points,density)), columns =['mid_points','density'])
    print(data)

#Question 1 part d
estimate_density(0.25, a, b)

#Question 1 part e
estimate_density(0.5, a, b)
    
#Question 1 part f
estimate_density(1, a, b)

#Question 1 part g
estimate_density(2, a, b)
#=======================================================================================================================
#Question 2
#Question 2 part a
dataset=pd.read_csv(r"C:\Users\Akash1313\Desktop\CS584_ML\Assignment1\NormalSample.csv",delimiter=',', usecols=["x"])
print(dataset.head())
print(dataset.describe())
data=list(dataset.x)
Q1,Q3= np.percentile(data, [25 ,75])
IQR=Q3-Q1
l_whisker=Q1-1.5*IQR 
u_whisker=Q3+1.5*IQR
print("Lower Whisker",l_whisker)
print("Upper Whisker",u_whisker)

#Question 2 part b
dataset=pd.read_csv(r"C:\Users\Akash1313\Desktop\CS584_ML\Assignment1\NormalSample.csv",delimiter=',')
new_dataset=dataset.groupby('group')
print(new_dataset.describe())
data_group_0=dataset[dataset.group == 0].x
data_group_1=dataset[dataset.group == 1].x

print("For Group 0:")
Q1_0,Q3_0= np.percentile(data_group_0,[25,75])
IQR = Q3_0-Q1_0
l_whisker_0=Q1_0-1.5*IQR 
u_whisker_0=Q3_0+1.5*IQR
print("\tLower Whisker",l_whisker_0)
print("\tUpper Whisker",u_whisker_0)

print("For Group 1:")
Q1_1,Q3_1= np.percentile(data_group_1,[25,75])
IQR = Q3_1-Q1_1
l_whisker_1=Q1_1-1.5*IQR 
u_whisker_1=Q3_1+1.5*IQR
print("\tLower Whisker",l_whisker_1)
print("\tUpper Whisker",u_whisker_1)

#Question 2 part c
plt.figure(figsize=(8,3))
sns.boxplot(x=dataset.x,color='gray')
plt.title("Box plot of x")
plt.show()

#Question 2 part d
new_data=pd.concat([dataset.x,data_group_0,data_group_1],axis=1, keys=['dataset.x','data_group_0','data_group_1'])
plt.figure(figsize=(8,8))
sns.boxplot(data=new_data,orient='h')
plt.title("Box plot of each group")
plt.show()

print("outliers of x for entire dataset are as follows:")
for pt in dataset.x:
    if pt < l_whisker or pt > u_whisker:
        print("\t",pt)

print("outliers of x for each group of data are as follows:")
print("\toutliers of x for group 0 are as follows:")
for pt in data_group_0:
    if pt < l_whisker_0 or pt>u_whisker_0:
        print("\t\t",pt)

print("\toutliers of x for group 1 are as follows:")
for pt in data_group_1:
    if pt < l_whisker_1 or pt>u_whisker_1:
        print("\t\t",pt)
#=======================================================================================================================
#Question 3
#Question 3 part a
dataset=pd.read_csv(r"C:\Users\Akash1313\Desktop\CS584_ML\Assignment1\Fraud.csv",delimiter=',')
print(dataset.head())
fraud = dataset[dataset.FRAUD == 1]
fraud_percentage=(len(fraud)/len(dataset))*100
print("Percent of investigations are found to be fraudulent is given by: ",round(fraud_percentage, 4))

#Question 3 part b
plt.figure(figsize=(12,6))
sns.boxplot(x="TOTAL_SPEND", y="FRAUD", data=dataset,orient='h')
plt.title("Box plot for Total amount of claims in dollars")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x="DOCTOR_VISITS", y="FRAUD", data=dataset,orient='h')
plt.title("Box plot for Number of visits to a doctor  ")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x="NUM_CLAIMS", y="FRAUD", data=dataset,orient='h')
plt.title("Box plot for Number of claims made recently")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x="MEMBER_DURATION", y="FRAUD", data=dataset,orient='h')
plt.title("Box plot for Membership duration in number of months")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x="OPTOM_PRESC", y="FRAUD", data=dataset,orient='h')
plt.title("Box plot for Number of optical examinations")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x="NUM_MEMBERS", y="FRAUD", data=dataset,orient='h')
plt.title("Box plot for Number of members covered")
plt.show()

#Question 3 part c
data=dataset.drop(['FRAUD','CASE_ID'],axis=1)
# Create a matrix x
x = np.matrix(data)
#Calculate the transpose of x
xtx = x.transpose() * x
print("\nt(x) * x = \n\t", xtx)
#Eigenvalue decomposition
evals, evecs = LA.eigh(xtx)
print("\nEigenvalues of x = \n\t", evals)
print("\nEigenvectors of x = \n\t", evecs)
# Here is the transformation matrix
transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)));
print("\nTransformation Matrix = \n\t", transf)
# Here is the transformed X
transf_x = x * transf;
print("\nThe Transformed x = \n\t", transf_x)
# Check columns of transformed X
xtx = transf_x.transpose() * transf_x;
print("\nExpect an Identity Matrix = \n\t", xtx)

#Question 3 part d
trainData = pd.DataFrame(transf_x)
target = dataset['FRAUD']
#Using NearestNeighbors
KNNSpec=NearestNeighbors(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs_=KNNSpec.fit(trainData)
#Using KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(trainData, target)
# Score Calculation:
_score = nbrs.score(trainData, target)
print("Score is:",_score)

#Question 3 part e
pd.set_option('display.expand_frame_repr', False)
focal = [[7500, 15, 3, 127, 2, 2]]
transf_focal = focal * transf;
myNeighbors_t = nbrs.kneighbors(transf_focal, return_distance = False)
print("My Neighbors = \n", myNeighbors_t)
print(dataset.iloc[list(myNeighbors_t[0])])

#Question 3 part f
class_prob = nbrs.predict_proba(transf_focal)
print(class_prob)
#=======================================================================================================================