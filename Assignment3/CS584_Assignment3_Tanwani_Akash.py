import sys
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
#Question 1
dataset =  pd.read_csv(r'C:/Users/Akash1313/Desktop/CS584_ML/Assignment3/claim_history.csv', delimiter=',')
y = dataset['CAR_USE']

dataset = dataset[['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']]
dataset['EDUCATION_VAL'] = dataset['EDUCATION'].map({'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})

train_x, test_x, train_y, test_y = train_test_split(dataset,y,train_size=0.75,test_size=0.25,random_state=60616,stratify=y)

#Question 1
print("\nFor Training Data:")
print('\tNumber of Observations = ', train_x.shape[0])
print('\tProportion of Dataset = ', (train_x.shape[0]/dataset.shape[0]))

print("\nFor Testing Data:")
print('\tNumber of Observations = ', test_x.shape[0])
print('\tProportion of Dataset = ', (test_x.shape[0]/dataset.shape[0]))

#Que1 part(a)
print("\nFor Training Data:")
print('Count of target variable is:\n',train_x.groupby("CAR_USE").size())
print('Proportion of target variable is:\n',(train_x.groupby("CAR_USE").size() / train_x.shape[0]))

#Question 1 part(b)
print("\nFor Testing Data:")
print('Count of target variable is:\n',test_x.groupby("CAR_USE").size())
print('Proportion of target variable is:\n',(test_x.groupby("CAR_USE").size() / test_x.shape[0]))

#Question 1 part(c)
P_c_g_train = train_x.groupby("CAR_USE").size()["Commercial"] / train_x.shape[0]
P_c_g_test = test_x.groupby("CAR_USE").size()["Commercial"] / test_x.shape[0] 
P_train_g_c = (P_c_g_train * 0.75) / ((P_c_g_train * 0.75) + (P_c_g_test * 0.25))
print('\nProbability(observation is in the Training partition / CAR_USE = Commercial) =',P_train_g_c)

#Question 1 part(d)
P_p_g_train = train_x.groupby("CAR_USE").size()["Private"] / train_x.shape[0]
P_p_g_test = test_x.groupby("CAR_USE").size()["Private"] / test_x.shape[0]
P_test_g_p = (P_p_g_test * 0.25) / ((P_p_g_train * 0.75) + (P_p_g_test * 0.25))
print('\nProbability(observation is in the Test partition / CAR_USE = Private) =',P_test_g_p,"\n")

#-------------------------------------------------------------------------------------------------------
#Question 2

def nominalEntropy(input_data,target):

    #possible combinations in the predictors(i.e,in nominal)
    column_vals = input_data.unique()
    lbranch = []
    #rbranch = []
    for i in range(1,(int(len(column_vals)/2))+1):
        lbranch1 = combinations(column_vals,i)
        for i in lbranch1:
            lbranch.append(list(i))
    #for element in lbranch:
    #    rbranch.append(list(set(column_vals).difference(element)))


    #finds the entropy split for each combination in the lbranch and gives the minimum entropy split of the column as the output
    minimum_entropy = sys.float_info.max
    minimum_subset1 = None
    minimum_subset2 = None
    minimum_table = None

    for combi in lbranch:
        dataTable = input_data.to_frame()
        dataTable['LE_Split'] = dataTable.iloc[:, 0].apply(lambda x: True if x in combi else False)
        crossTable = pd.crosstab(index=dataTable['LE_Split'], columns=target, margins=True, dropna=True)

        n_rows = crossTable.shape[0]
        n_columns = crossTable.shape[1]

        table_entropy = 0
        for i_row in range(n_rows - 1):
            row_entropy = 0
            for i_column in range(n_columns):
                proportion = crossTable.iloc[i_row, i_column] / crossTable.iloc[i_row, (n_columns - 1)]
                if proportion > 0:
                    row_entropy -= proportion * np.log2(proportion)
            # print('Row = ', i_row, 'Entropy =', row_entropy)
            # print(' ')
            table_entropy += row_entropy * crossTable.iloc[i_row, (n_columns - 1)]
        table_entropy = table_entropy / crossTable.iloc[(n_rows - 1), (n_columns - 1)]

        if table_entropy < minimum_entropy:
            minimum_entropy = table_entropy
            minimum_subset1 = combi
            minimum_subset2 = list(set(column_vals).difference(combi))
            minimum_table = crossTable
    return minimum_table,minimum_entropy,minimum_subset1,minimum_subset2


def ordinalEntropy(input_data,target):
    interval_vals = sorted(input_data.unique()) #[0, 1, 2, 3, 4]
    minimum_entropy = sys.float_info.max
    minimum_subset1 = None
    minimum_subset2 = None
    minimum_table = None

    for i in range(interval_vals[0], interval_vals[len(interval_vals) - 1]):
        dataTable = input_data.to_frame()
        dataTable['LE_Split'] = dataTable.iloc[:, 0] <= i + 0.5
        crossTable = pd.crosstab(index=dataTable['LE_Split'], columns=target, margins=True, dropna=True)

        n_rows = crossTable.shape[0]
        n_columns = crossTable.shape[1]

        table_entropy = 0
        for i_row in range(n_rows - 1):
            row_entropy = 0
            for i_column in range(n_columns):
                proportion = crossTable.iloc[i_row, i_column] / crossTable.iloc[i_row, (n_columns - 1)]
                if proportion > 0:
                    row_entropy -= proportion * np.log2(proportion)
            # print('Row = ', i_row, 'Entropy =', row_entropy)
            # print(' ')
            table_entropy += row_entropy * crossTable.iloc[i_row, (n_columns - 1)]
        table_entropy = table_entropy / crossTable.iloc[(n_rows - 1), (n_columns - 1)]

        if table_entropy < minimum_entropy:
            minimum_entropy = table_entropy
            minimum_interval = i + 0.5
            minimum_table = crossTable
    return minimum_table,minimum_entropy,minimum_interval


#Que2 part(a)
root_Entropy = 0
target_val = train_y.unique()
for val in target_val:
    temp = len(train_y.loc[train_y == val])/len(train_y)
    root_Entropy += -( temp * np.log2(temp) )
print('Entropy value of the root node is:',root_Entropy,"\n")

#Que2 part(b)
#For Layer 0 Evaluate entropy of each column to see the minimum 
c_type_table,c_type_entropy,c_type_subset1,c_type_subset2 = nominalEntropy(train_x['CAR_TYPE'],train_y)
occu_table,occu_entropy,occu_subset1,occu_subset2 = nominalEntropy(train_x['OCCUPATION'],train_y)
edu_table,edu_entropy,edu_interval = ordinalEntropy(train_x['EDUCATION_VAL'],train_y)
print("\nEntropy of CarType is :",c_type_entropy)
print("Entropy of Occupation is :",occu_entropy)
print("Entropy of Education is :",edu_entropy,"\n")
print('Best split : "OCCUPATION"')
print('the values in the two branches are:')
print('For Left branch: ',occu_subset1)
print('For Right branch: ',occu_subset2)

#Que2 part(c)
print('\nFor the first layer Entropy of the split of will be:',occu_entropy)
print('Occupation contingency table: ')
print(occu_table,"\n")

#Que2 part(e)
#Layer 1: 
train_x_left = train_x[train_x['OCCUPATION'].isin(occu_subset1)]
train_y_left = train_x_left['CAR_USE']

train_x_right = train_x[train_x['OCCUPATION'].isin(occu_subset2)]
train_y_right = train_x_right['CAR_USE']

#Layer 1: Left node
c_type_table_l,c_type_entropy_l,c_type_subset1_l,c_type_subset2_l = nominalEntropy(train_x_left['CAR_TYPE'],train_y_left)
occu_table_l,occu_entropy_l,occu_subset1_l,occu_subset2_l = nominalEntropy(train_x_left['OCCUPATION'],train_y_left)
edu_table_l,edu_entropy_l,edu_interval_l = ordinalEntropy(train_x_left['EDUCATION_VAL'],train_y_left)
#print(c_type_table_l,c_type_entropy_l,c_type_subset1_l,c_type_subset2_l)
#print(occu_table_l,occu_entropy_l,occu_subset1_l,occu_subset2_l)
#print(edu_table_l,edu_entropy_l,edu_interval_l)

#Layer 1: Right node
c_type_table_r,c_type_entropy_r,c_type_subset1_r,c_type_subset2_r = nominalEntropy(train_x_right['CAR_TYPE'],train_y_right)
occu_table_r,occu_entropy_r,occu_subset1_r,occu_subset2_r = nominalEntropy(train_x_right['OCCUPATION'],train_y_right)
edu_table_r,edu_entropy_r,edu_interval_r = ordinalEntropy(train_x_right['EDUCATION_VAL'],train_y_right)
#print(c_type_table_r,c_type_entropy_r,c_type_subset1_r,c_type_subset2_r)
#print(occu_table_r,occu_entropy_r,occu_subset1_r,occu_subset2_r)
#print(edu_table_r,edu_entropy_r,edu_interval_r)

#leaves
train_x_left_left = train_x_left[train_x_left['EDUCATION_VAL'] <= edu_interval_l]
train_x_left_right = train_x_left[train_x_left['EDUCATION_VAL'] > edu_interval_l]

train_x_right_left = train_x_right[train_x_right['CAR_TYPE'].isin(c_type_subset1_r)]
train_x_right_right = train_x_right[train_x_right['CAR_TYPE'].isin(c_type_subset2_r)]

print('\nLeaf One:')
print('Decision rules are: ',occu_subset1,['Below High School'])
print('Counts: ')
table_leaf1 = train_x_left_left['CAR_USE'].value_counts().to_frame().reset_index()
table_leaf1.columns = ['CAR_USE','COUNT']
table_leaf1['PROPORTION'] = table_leaf1['COUNT'] / train_x_left_left.shape[0]
print(table_leaf1)
entropy_val = 0
target_val = train_x_left_left['CAR_USE'].unique()
for value in target_val:
    temp = len(train_x_left_left.loc[train_x_left_left['CAR_USE'] == value])/len(train_x_left_left['CAR_USE'])
    entropy_val += -( temp * np.log2(temp) )
print('Entropy:',entropy_val,'\n')

print('\nLeaf Two:')
print('Decision rules are : ',occu_subset1,['High School', 'Bachelors', 'Masters', 'Doctors'])
print('Counts: ')
table_leaf2 = train_x_left_right['CAR_USE'].value_counts().to_frame().reset_index()
table_leaf2.columns = ['CAR_USE','COUNT']
table_leaf2['PROPORTION'] = table_leaf2['COUNT'] / train_x_left_right.shape[0]
print(table_leaf2)
entropy_val = 0
target_val = train_x_left_right['CAR_USE'].unique()
for value in target_val:
    temp = len(train_x_left_right.loc[train_x_left_right['CAR_USE'] == value])/len(train_x_left_right['CAR_USE'])
    entropy_val += -( temp * np.log2(temp) )
print('Entropy:',entropy_val,'\n')

print('\nLeaf Three:')
print('Decision rules are: ',occu_subset2,c_type_subset1_r)
print('Counts: ')
table_leaf3 = train_x_right_left['CAR_USE'].value_counts().to_frame().reset_index()
table_leaf3.columns = ['CAR_USE','COUNT']
table_leaf3['PROPORTION'] = table_leaf3['COUNT'] / train_x_right_left.shape[0]
print(table_leaf3)
entropy_val = 0
target_val = train_x_right_left['CAR_USE'].unique()
for value in target_val:
    temp = len(train_x_right_left.loc[train_x_right_left['CAR_USE'] == value])/len(train_x_right_left['CAR_USE'])
    entropy_val += -( temp * np.log2(temp) )
print('Entropy:',entropy_val,'\n')

print('\nLeaf Four:')
print('Decision rules are ',occu_subset2,c_type_subset2_r)
print('Counts: ')
table_leaf4 = train_x_right_right['CAR_USE'].value_counts().to_frame().reset_index()
table_leaf4.columns = ['CAR_USE','COUNT']
table_leaf4['PROPORTION'] = table_leaf4['COUNT'] / train_x_right_right.shape[0]
print(table_leaf4)
entropy_val = 0
target_val = train_x_right_right['CAR_USE'].unique()
for value in target_val:
    temp = len(train_x_right_right.loc[train_x_right_right['CAR_USE'] == value])/len(train_x_right_right['CAR_USE'])
    entropy_val += -( temp * np.log2(temp) )
print('Entropy:',entropy_val,'\n')

#------------------------------------------------------------------------------------------
#These functions are used in que 2F part and que 3
def predict_class(input_data):
    if input_data['OCCUPATION'] in ('Blue Collar', 'Student', 'Unknown'):
        if input_data['EDUCATION_VAL'] <= 0.5:
            return [0.269355, 0.730645]
        else:
            return [0.837659, 0.162341]
    else:
        if input_data['CAR_TYPE'] in ('Minivan', 'SUV', 'Sports Car'):
            return [0.00842, 0.99158]
        else:
            return [0.534197, 0.465803]

#threshold Calculation:
threshold = train_x.groupby("CAR_USE").size()["Commercial"] / train_x.shape[0]

# predict probability for testing data
predicted_p_y = np.ndarray(shape=(len(test_x), 2), dtype=float)
counter = 0
input_data=test_x
for index, row in input_data.iterrows():
    probability = predict_class(input_data=row)
    predicted_p_y[counter] = probability
    counter += 1
predicted_p_y = predicted_p_y[:, 0]

#-----------------------------------------------------------------------------------------
#Que 2 part(f)
fpr, tpr, thresholds = metrics.roc_curve(test_y, predicted_p_y, pos_label='Commercial')
cutoff = np.where(thresholds > 1.0, np.nan, thresholds)
plt.plot(cutoff, tpr, marker = 'o', label = 'True Positive',color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(cutoff, fpr, marker = 'o', label = 'False Positive',color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True, fontsize = 'small')
plt.show()

ks_Statistic = 0
ks_Threshold = 0
for i in range(len(thresholds)):
    if tpr[i]-fpr[i]>ks_Statistic:
        ks_Statistic = tpr[i]-fpr[i]
        ks_Threshold = thresholds[i]
print("\nThe Kolmogorov Smirnov statistic is ",ks_Statistic)
print("Event probability cutoff value",ks_Threshold)

#-------------------------------------------------------------------------------------------------------
#Question 3
#Functions are defined above Que2 F part
test_x=test_x[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]
test_x['EDUCATION'] = test_x['EDUCATION'].map({'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})

# determine the predicted class
pred_y = np.empty_like(test_y)
for i in range(test_y.shape[0]):
    if predicted_p_y[i] > threshold:
        pred_y[i] = 'Commercial'
    else:
        pred_y[i] = 'Private'
        
#Que 3 part(a)    
print("\nthreshold value is:",threshold)
accuracy = metrics.accuracy_score(test_y, pred_y)
misclassification_rate = 1 - accuracy
print('the Misclassification Rate in the Test partition is:',misclassification_rate)

#Que 3 part(b)
pred_yy = np.empty_like(test_y)
for i in range(test_y.shape[0]):
    if predicted_p_y[i] > ks_Threshold:
        pred_yy[i] = 'Commercial'
    else:
        pred_yy[i] = 'Private'

ks_accuracy = metrics.accuracy_score(test_y, pred_yy)
ks_misclassification_rate = 1 - ks_accuracy
print('\nUsing Kolmogorov-Smirnov event probability cutoff value the Misclassification Rate in the Test partition is:',ks_misclassification_rate)

#Que 3 part(c)
RASE = 0.0
for y, p_p_y in zip(test_y, predicted_p_y):
    if y == 'Commercial':
        RASE += (1 - p_p_y) ** 2
    else:
        RASE += (0 - p_p_y) ** 2
RASE = np.sqrt(RASE / test_y.shape[0])
print('\nthe Root Average Squared Error in the Test partition is:',RASE)

#Que 3 part(d)
true_y = 1.0 * np.isin(test_y, ['Commercial'])
Events = []
NonEvents = []

for i in range(len(true_y)):
    if true_y[i]==1:
        Events.append(predicted_p_y[i])
    else:
        NonEvents.append(predicted_p_y[i])

ConcordantPairs = 0
DiscordantPairs = 0
TiedPairs = 0

for i in Events:
    for j in NonEvents:
        if i>j:
            ConcordantPairs = ConcordantPairs + 1
        elif i<j:
            DiscordantPairs = DiscordantPairs + 1
        else:
            TiedPairs = TiedPairs + 1

AUC = 0.5 + 0.5 * ((ConcordantPairs-DiscordantPairs)/(ConcordantPairs+DiscordantPairs+TiedPairs))
print('\nthe Area Under Curve in the Test partition is:',AUC)

#Que 3 part(e)
#GINI=2* AUC-1
GINI=((ConcordantPairs-DiscordantPairs)/(ConcordantPairs+DiscordantPairs+TiedPairs))
print('\nthe Gini Coefficient in the Test partition is:',GINI)

#Que 3 part(f)
GC = ((ConcordantPairs-DiscordantPairs)/(ConcordantPairs+DiscordantPairs))
print("\nthe Goodman-Kruskal Gamma statistic in the Test partition is :",GC)

#Que 3 part(g)
one_minus_specificity, sensitivity, thresholds = metrics.roc_curve(test_y, predicted_p_y, pos_label='Commercial')
one_minus_specificity = np.append([0], one_minus_specificity)
sensitivity = np.append([0], sensitivity)
one_minus_specificity = np.append(one_minus_specificity, [1])
sensitivity = np.append(sensitivity, [1])
# plot the roc curve
plt.figure(figsize=(6, 6))
plt.plot(one_minus_specificity, sensitivity, marker='o', color='blue', linestyle='solid', linewidth=2, markersize=6)
plt.plot([0, 1], [0, 1], color='red', linestyle=':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()





