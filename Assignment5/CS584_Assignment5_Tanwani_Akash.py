# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:32:22 2020

@author: Akash1313
"""
#Question1
import pandas as pd
import numpy as np
import sklearn.neural_network as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


print()
print("------------------------Ques1-------------------------------")
print()
dataset = pd.read_csv(r'C:/Users/Akash1313/Desktop/CS584_ML/Assignment5/SpiralWithCluster.csv', delimiter=',',
                                      usecols=['x', 'y', 'SpectralCluster'])

print("------------------------Ques1(a)-------------------------------")
numObsSC1=(dataset['SpectralCluster'].value_counts()[1] / len(dataset))
print('Percent of the observations have SpectralCluster equals to 1 are: ',100 * numObsSC1,"% ")

print("\n------------------------Ques1(b)-------------------------------")
def Build_MLP (func,layer,neuron,flag=0):

    # Build Neural Network
    nn_obj = nn.MLPClassifier(hidden_layer_sizes=(neuron,) * layer, activation=func, verbose=False,solver='lbfgs', learning_rate_init=0.1, max_iter=5000, random_state=20200408)
    this_fit = nn_obj.fit(dataset[['x', 'y']], dataset[['SpectralCluster']])
    y_predprob = nn_obj.predict_proba(dataset[['x', 'y']])
    n_iter = nn_obj.n_iter_
    loss = nn_obj.loss_

    #predict class by considering numObsSC1 as a threshold value:
    pred_y = np.where(y_predprob[:,1] >= numObsSC1, 1, 0)
    #misclassification rate calculation:
    accuracy = metrics.accuracy_score(dataset[['SpectralCluster']], pred_y)
    miscls_rate = 1 - accuracy
    if flag==1:
        dataset['y_pred_0'] = y_predprob[:, 0]
        dataset['y_pred_1'] = y_predprob[:, 1]
        dataset['_PredictedClass_'] = pred_y
        act_func=nn_obj.out_activation_
        return (n_iter,loss,accuracy, miscls_rate,act_func)
    return (n_iter,loss, miscls_rate)

activation_function = ['identity', 'logistic', 'relu', 'tanh']
num_layer = range(1, 6, 1)
num_neuron = range(1, 11, 1)
res_table = pd.DataFrame(columns=['Index', 'ActivationFunction', 'nLayers', 'nNeuronsPerLayer', 'nIterations', 'Loss','MisclassificationRate'])
index = 0
pred_y = np.empty_like(dataset[['SpectralCluster']])

for func in activation_function:
    min_loss = min_miscls_rate = float("inf")
    nLayer= nNeuron= nIter= -1
    for layer in num_layer:
        for neuron in num_neuron:
            n_iter,loss, miscls_rate = Build_MLP(func,layer,neuron)
            # find neural network with minimum loss and misclassification rate
            if loss <= min_loss and miscls_rate <= min_miscls_rate:
                min_loss = loss
                min_miscls_rate = miscls_rate
                nLayer = layer
                nNeuron = neuron
                nIter = n_iter

    res_table = res_table.append(pd.DataFrame([[index, func, nLayer, nNeuron, nIter, min_loss, min_miscls_rate]],
                         columns=['Index', 'ActivationFunction', 'nLayers', 'nNeuronsPerLayer','nIterations', 'Loss', 'MisclassificationRate']))
    index += 1
res_table = res_table.set_index('Index')
pd.set_option('display.max_columns', 10)
pd.set_option('display.expand_frame_repr', False)
print(res_table)

min_loss = min_miscls_rate = float("inf")
index = None
for ind, row in res_table.iterrows():
    if row['Loss'] <= min_loss and row['MisclassificationRate'] <= min_miscls_rate:
        index = ind
        min_loss = row['Loss']
        min_miscls_rate = row['MisclassificationRate']

#Neural network with minimum loss and misclassification rate:
func=res_table.loc[index]['ActivationFunction']
layer=res_table.loc[index]['nLayers']
neuron=res_table.loc[index]['nNeuronsPerLayer']
n_iter,loss,accuracy,miscls_rate,act_func, = Build_MLP(func,layer,neuron,flag=1)

print("\n------------------------Ques1(c)-------------------------------")
print('The activation function for the output layer is ',act_func)

print("\n------------------------Ques1(d)-------------------------------")
print(res_table.loc[index])

print("\n------------------------Ques1(e)-------------------------------")
plt.figure(figsize=(8,8))
carray = ['red', 'blue']
for i in range(2):
    subData=dataset[dataset['_PredictedClass_'] == i]
    plt.scatter(subData['x'], subData['y'], c=carray[i], label=i)
plt.xlabel('coordinates of x axis')
plt.ylabel('coordinates of y axis')
plt.title('Scatter Plot of x and y coordinates for MLP (relu, 4 Layers, 10 Neurons)')
plt.legend(title='Predicted_Cluster', loc='best')
plt.grid(True)
plt.show()

print("------------------------Ques1(f)-------------------------------")
pd.set_option('float_format', '{:.10f}'.format)
print(dataset[dataset['_PredictedClass_'] == 1]['y_pred_1'].describe())

#Question2
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import sklearn.svm as svm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print()
print("------------------------Ques2-------------------------------")
print()
dataset = pd.read_csv(r'C:/Users/Akash1313/Desktop/CS584_ML/Assignment5/SpiralWithCluster.csv', delimiter=',',
                                      usecols=['x', 'y', 'SpectralCluster'])

xTrain = dataset[['x', 'y']]
yTrain = dataset['SpectralCluster']

svm_Model = svm.SVC(kernel='linear', decision_function_shape='ovr', random_state=20200408, max_iter=-1)
thisFit = svm_Model.fit(xTrain, yTrain)

print("------------------------Ques2(a)-------------------------------")
print('Equation of the separating hyperplane is given by: "𝑤_0 + 𝐰_1*X + w_2*Y =𝟎"\nHere')
print('Intercept= w_0=',np.round(thisFit.intercept_[0],7))
print('Coefficients are \n\t w_1= ',np.round(thisFit.coef_[0][0], 7),'\n\t w_2= ',np.round(thisFit.coef_[0][1], 7))
print('Therefore the Equation will be :\n\t(',np.round(thisFit.intercept_[0],7),')+(',np.round(thisFit.coef_[0][0], 7),'*X)+(',
      np.round(thisFit.coef_[0][1], 7),'*Y)=𝟎')

print("------------------------Ques2(b)-------------------------------")
y_predictClass = thisFit.predict(xTrain)
accuracy = metrics.accuracy_score(yTrain, y_predictClass)
miscls_rate = 1 - accuracy
print('Misclassification rate of the model is: ',miscls_rate)

print("------------------------Ques2(c)-------------------------------")
dataset['_PredictedClass_'] = y_predictClass
xx = np.linspace(-5, 5)
yy = np.zeros((len(xx), 1))
for j in range(1):
    w = thisFit.coef_[j, :]
    a = -w[0] / w[1]
    yy[:, j] = a * xx - (thisFit.intercept_[j]) / w[1]

# plot 
plt.figure(figsize=(8,8))
carray = ['red', 'blue']
for i in range(2):
    subData=dataset[dataset['_PredictedClass_'] == i]
    plt.scatter(subData['x'], subData['y'], c=carray[i], label=i)

plt.plot(xx, yy[:, 0], color='black', linestyle='--')
plt.xlabel('coordinates of x axis')
plt.ylabel('coordinates of y axis')
plt.title('SVM Scatter plot of x and y coordinates')
plt.legend(title='Predicted_Cluster', loc='best')
plt.grid(True)
plt.show()

print("------------------------Ques2(d)-------------------------------")
def customArcTan(z):
    theta = np.where(z < 0.0, 2.0 * np.pi + z, z)
    return (theta)

# get radius and theta coordinates
dataset['radius'] = np.sqrt(dataset['x'] ** 2 + dataset['y'] ** 2)
dataset['theta'] = np.arctan2(dataset['y'], dataset['x']).apply(customArcTan)

# plot 
plt.figure(figsize=(8,8))
carray = ['red', 'blue']
for i in range(2):
    subData = dataset[dataset['SpectralCluster'] == i]
    plt.scatter(subData['radius'], subData['theta'], c=carray[i], label=i)
plt.xlabel('Radius-Coordinates')
plt.ylabel('Theta-Coordinates')
plt.ylim(-1, 7)
plt.title('Scatter plot of polar coordinates of x and y')
plt.legend(title='Spectral_Cluster', loc='best')
plt.grid(True)
plt.show()


print("------------------------Ques2(e)-------------------------------")
dataset['Group'] = 2

dataset.loc[((dataset['theta'] > 3.00) & (dataset['radius'] < 2)) | ((dataset['theta'] > 4.00) & (dataset['radius'] < 3)), 'Group'] = 1
dataset.loc[(dataset['theta'] > 6.00) & (dataset['radius'] < 1.5), 'Group'] = 0
dataset.loc[((dataset['theta'] <= 2.00) & (dataset['radius'] > 2.5))| ((dataset['theta'] < 3.5) & (dataset['radius'] > 3.0)), 'Group'] = 3

# plot 
plt.figure(figsize=(8,8))
carray = ['red', 'blue', 'green', 'black']
for i in range(4):
    subData = dataset[dataset['Group'] == i]
    plt.scatter(x=subData['radius'], y=subData['theta'], c=carray[i], label=i)
plt.xlabel('Radius-Coordinates')
plt.ylabel('Theta-Coordinates')
plt.title('Scatter plot of Four Segments')
plt.legend(title='Group', loc='best', )
plt.grid(True)
plt.show()

print("------------------------Ques2(f)-------------------------------")
#SVM 0: using Group 0 vs Group 1
dataset0=dataset[dataset['Group'].isin([0,1])]
xTrain0= dataset0[['radius', 'theta']]
yTrain0=dataset0['Group']
svm_Model0 = svm.SVC(kernel="linear", random_state=20200408, decision_function_shape='ovr', max_iter=-1)
thisFit0=svm_Model0.fit(xTrain0,yTrain0)

print('Equation of the separating hyperplane is given by: "𝑤_0 + 𝐰_1*X + w_2*Y =𝟎"\n\n For SVM 0:')
print('Intercept= w_0=',np.round(thisFit0.intercept_[0],7))
print('Coefficients are \n\t w_1= ',np.round(thisFit0.coef_[0][0], 7),'\n\t w_2= ',np.round(thisFit0.coef_[0][1], 7))
print('Therefore the Equation will be :\n\t(',np.round(thisFit0.intercept_[0],7),')+(',np.round(thisFit0.coef_[0][0], 7),'*X)+(',
      np.round(thisFit0.coef_[0][1], 7),'*Y)=𝟎')

w = thisFit0.coef_[0]
a = -w[0] / w[1]
xx0 = np.linspace(1, 4)
yy0 = a * xx0 - (thisFit0.intercept_[0]) / w[1]


#SVM 1: using Group 1 vs Group 2
dataset1=dataset[dataset['Group'].isin([1,2])]
xTrain1= dataset1[['radius', 'theta']]
yTrain1=dataset1['Group']
svm_Model1 = svm.SVC(kernel="linear", random_state=20200408, decision_function_shape='ovr', max_iter=-1)
thisFit1=svm_Model1.fit(xTrain1,yTrain1)

print('\n For SVM 1:')
print('Intercept= w_0=',np.round(thisFit1.intercept_[0],7))
print('Coefficients are \n\t w_1= ',np.round(thisFit1.coef_[0][0], 7),'\n\t w_2= ',np.round(thisFit1.coef_[0][1], 7))
print('Therefore the Equation will be :\n\t(',np.round(thisFit1.intercept_[0],7),')+(',np.round(thisFit1.coef_[0][0], 7),'*X)+(',
      np.round(thisFit1.coef_[0][1], 7),'*Y)=𝟎')

w = thisFit1.coef_[0]
a = -w[0] / w[1]
xx1 = np.linspace(1, 4)
yy1 = a * xx1 - (thisFit1.intercept_[0]) / w[1]

#SVM 2: using Group 2 vs Group 3
dataset2=dataset[dataset['Group'].isin([2,3])]
xTrain2= dataset2[['radius', 'theta']]
yTrain2=dataset2['Group']
svm_Model2 = svm.SVC(kernel="linear", random_state=20200408, decision_function_shape='ovr', max_iter=-1)
thisFit2=svm_Model2.fit(xTrain2,yTrain2)

print('\n For SVM 2:')
print('Intercept= w_0=',np.round(thisFit2.intercept_[0],7))
print('Coefficients are \n\t w_1= ',np.round(thisFit2.coef_[0][0], 7),'\n\t w_2= ',np.round(thisFit2.coef_[0][1], 7))
print('Therefore the Equation will be :\n\t(',np.round(thisFit2.intercept_[0],7),')+(',np.round(thisFit2.coef_[0][0], 7),'*X)+(',
      np.round(thisFit2.coef_[0][1], 7),'*Y)=𝟎')

w = thisFit2.coef_[0]
a = -w[0] / w[1]
xx2 = np.linspace(1, 4)
yy2 = a * xx2 - (thisFit2.intercept_[0]) / w[1]

print("------------------------Ques2(g)-------------------------------")
plt.figure(figsize=(8,8))
carray = ['red', 'blue', 'green', 'black']
for i in range(4):
    subData = dataset[dataset['Group'] == i]
    plt.scatter(x=subData['radius'], y=subData['theta'], c=carray[i], label=i)
plt.plot(xx0, yy0, color = 'black', linestyle = '-')
plt.plot(xx1, yy1, color = 'black', linestyle = '-')
plt.plot(xx2, yy2, color = 'black', linestyle = '-')
plt.xlabel('Radius-Coordinates')
plt.ylabel('Theta-Coordinates')
plt.title('Scatterplot of polar coordinates of x & y along with hyperplanes')
plt.legend(title='Group', loc='best', )
plt.grid(True)
plt.show()


print("------------------------Ques2(h)-------------------------------")
h0_xx = xx0 * np.cos(yy0)
h0_yy = xx0 * np.sin(yy0)
h1_xx = xx1 * np.cos(yy1)
h1_yy = xx1 * np.sin(yy1)
h2_xx = xx2 * np.cos(yy2)
h2_yy = xx2 * np.sin(yy2)

plt.figure(figsize=(8,8))
carray = ['red', 'blue']
for i in range(2):
    subdata = dataset[dataset['SpectralCluster'] == i]
    plt.scatter(subdata['x'], subdata['y'], c=carray[i], label=i)
    
plt.plot(h0_xx, h0_yy, color = 'red', linestyle = '--')
plt.plot(h1_xx, h1_yy, color = 'black', linestyle = '--')
plt.plot(h2_xx, h2_yy, color = 'black', linestyle = '--')
plt.title('Scatterplot of cartesian coordinates of x & y along with hyperplanes')
plt.xlabel('coordinates of x axis')
plt.ylabel('coordinates of y axis')
plt.legend(title='Pedicted class')
plt.grid(True)
plt.show()