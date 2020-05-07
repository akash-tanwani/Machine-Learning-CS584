# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:12:36 2020

@author: Akash1313
"""
#Question 1
###############################################################################
print("---------------------------Question1----------------------------------------")

import numpy as np
import pandas as pd
import scipy as sp
import sympy
import statsmodels.api as stats
import warnings

warnings.filterwarnings('ignore')

def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)
    
    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)
        print("-------------------------------ans 1a------------------------------------------")
        aliased_indices = [x for x in range(nFullParam) if (x not in inds)]
        aliased_params = [fullX.columns[x] for x in aliased_indices]
        print("the aliased columns in our model matrix are:\n ")
        for i in aliased_params:
            print(i)

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    #if (debug == 'Y'):
    print(thisFit.summary())
    print("Model Parameter Estimates:\n", thisParameter)
    print("Model Log-Likelihood Value =", thisLLK)
    print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pd.DataFrame(np.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pd.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)


#Que 1
dataframe = pd.read_csv(r'C:/Users/Akash1313/Desktop/CS584_ML/Assignment4/Purchase_Likelihood.csv', delimiter=',')
dataframe = dataframe.dropna()

y = dataframe['insurance'].astype('category')

xGS = pd.get_dummies(dataframe[['group_size']].astype('category'))
xH = pd.get_dummies(dataframe[['homeowner']].astype('category'))
xMC = pd.get_dummies(dataframe[['married_couple']].astype('category'))

# Intercept only model
designX = pd.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'N')

# Intercept + GS
print("\n\n---------------------GS------------------------------------")
designX = stats.add_constant(xGS, prepend=True)
LLK_1GS, DF_1GS, fullParams_1GS = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1GS - LLK0)
testDF = DF_1GS - DF0
testPValue = sp.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('Degreee of Freedom = ', testDF)
print('      Significance = ', testPValue)
print('Feature Importance Index = ', -np.log10(testPValue))

# Intercept + GS + H
print("\n\n--------------Intercept + GS + H----------------------------")
designX = xGS
designX = designX.join(xH)
designX = stats.add_constant(designX, prepend=True)
LLK_1GS_1H, DF_1GS_1H, fullParams_1GS_HJ = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1GS_1H - LLK_1GS)
testDF = DF_1GS_1H - DF_1GS
testPValue = sp.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print('Feature Importance Index = ', -np.log10(testPValue))

# Intercept + GS + H + MC
print("\n\n--------------Intercept + GS + H + MC----------------------------")
designX = xGS
designX = designX.join(xH)
designX = designX.join(xMC)
designX = stats.add_constant(designX, prepend=True)
LLK_1GS_1H_1MC, DF_1GS_1H_1MC, fullParams_1GS_1H_1MC = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1GS_1H_1MC - LLK_1GS_1H)
testDF = DF_1GS_1H_1MC - DF_1GS_1H
testPValue = sp.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print('Feature Importance Index = ', -np.log10(testPValue))

# Create the columns for the group_size * homeowner interaction effect
xGS_H = create_interaction(xGS, xH)

# Intercept + GS + H + MC + GS * H
print("\n\n--------------Intercept + GS + H + MC + GS * H----------------------------")
designX = xGS
designX = designX.join(xH)
designX = designX.join(xMC)
designX = designX.join(xGS_H)
designX = stats.add_constant(designX, prepend=True)
LLK_2GS_H, DF_2GS_H, fullParams_2GS_H = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_2GS_H - LLK_1GS_1H_1MC)
testDF = DF_2GS_H - DF_1GS_1H_1MC
testPValue = sp.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print('Feature Importance Index = ', -np.log10(testPValue))

# Create the columns for the e.	group_size * married_couple interaction effect
xGS_MC = create_interaction(xGS, xMC)

# Intercept + GS + H + MC + GS * H + GS * MC
print("\n\n--------------Intercept + GS + H + MC + GS * H + GS * MC----------------------------")
designX = xGS
designX = designX.join(xH)
designX = designX.join(xMC)
designX = designX.join(xGS_H)
designX = designX.join(xGS_MC)
designX = stats.add_constant(designX, prepend=True)
LLK_2GS_MC, DF_2GS_MC, fullParams_2GS_MC = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_2GS_MC - LLK_2GS_H)
testDF = DF_2GS_MC - DF_2GS_H
testPValue = sp.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print('Feature Importance Index = ', -np.log10(testPValue))

# Create the columns for the e.	homeowner * married_couple interaction effect
xH_MC = create_interaction(xH, xMC)

# Intercept + GS + H + MC + GS * H + GS * MC + H * MC
print("\n\n--------------Intercept + GS + H + MC + GS * H + GS * MC + H * MC----------------------------")
designX = xGS
designX = designX.join(xH)
designX = designX.join(xMC)
designX = designX.join(xGS_H)
designX = designX.join(xGS_MC)
designX = designX.join(xH_MC)
designX = stats.add_constant(designX, prepend=True)
LLK_2H_MC, DF_2H_MC, fullParams_2H_MC = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2H_MC - LLK_2GS_MC)
testDF = DF_2H_MC - DF_2GS_MC
testPValue = sp.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print('Feature Importance Index = ', -np.log10(testPValue))

print("-------------------------------ans 1b------------------------------------------")
print("degrees of freedom for our model is",testDF)
print()

#Question 2
###############################################################################
print("---------------------------Question2----------------------------------------")
# Build a multionomial logistic model
logit = stats.MNLogit(y, designX)
this_fit = logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)

data = []
for i in range(1, 5):
    for j in range(2):
        for k in range(2):
            data.append([i, j, k])

dataset = pd.DataFrame(data, columns=['group_size', 'homeowner', 'married_couple'])
xGS = pd.get_dummies( dataset[['group_size']].astype('category'))
xH = pd.get_dummies(dataset[['homeowner']].astype('category'))
xMC = pd.get_dummies(dataset[['married_couple']].astype('category'))
xGS_H = create_interaction(xGS, xH)
xGS_MC = create_interaction(xGS, xMC)
xH_MC = create_interaction(xH, xMC)

designX = xGS
designX = designX.join(xH)
designX = designX.join(xMC)
designX = designX.join(xGS_H)
designX = designX.join(xGS_MC)
designX = designX.join(xH_MC)
designX = stats.add_constant(designX, prepend=True)

insurance_pred = this_fit.predict(exog = designX)
insurance_result=pd.concat([dataset, insurance_pred],axis=1)
print("-------------------------------ans 2a------------------------------------------")
print(insurance_pred)

print("-------------------------------ans 2b------------------------------------------")
insurance_result['oddVal(prob_I1/prob_I0)'] = insurance_result[1] / insurance_result[0]
print(insurance_result[['group_size','homeowner','married_couple','oddVal(prob_I1/prob_I0)']])
max_row = insurance_result.loc[insurance_result['oddVal(prob_I1/prob_I0)'].idxmax()]
print("The maximum odd value is obtained when \ngroup_size =",max_row['group_size'],", homeowner = ",max_row['homeowner'],", married_couple = ",max_row['married_couple'])
print('The maximum odd value is: ',max_row['oddVal(prob_I1/prob_I0)'])   

print("-------------------------------ans 2c------------------------------------------")
prob_In2_GS3 = (dataframe[dataframe['group_size']==3].groupby('insurance').size()[2]/dataframe[dataframe['group_size']==3].shape[0])
prob_In0_GS3 = (dataframe[dataframe['group_size']==3].groupby('insurance').size()[0]/dataframe[dataframe['group_size']==3].shape[0])
odds1 = prob_In2_GS3/prob_In0_GS3

prob_In2_GS1 = (dataframe[dataframe['group_size']==1].groupby('insurance').size()[2]/dataframe[dataframe['group_size']==1].shape[0])
prob_In0_GS1 = (dataframe[dataframe['group_size']==1].groupby('insurance').size()[0]/dataframe[dataframe['group_size']==1].shape[0])
odds2 = prob_In2_GS1/prob_In0_GS1

oddsRatio = odds1/odds2
print(oddsRatio)

print("-------------------------------ans 2d------------------------------------------")
prob_In0_H1 = (dataframe[dataframe['homeowner']==1].groupby('insurance').size()[0]/dataframe[dataframe['homeowner']==1].shape[0])
prob_In1_H1 = (dataframe[dataframe['homeowner']==1].groupby('insurance').size()[1]/dataframe[dataframe['homeowner']==1].shape[0])
odds1 = prob_In0_H1/prob_In1_H1

prob_In0_H0 = (dataframe[dataframe['homeowner']==0].groupby('insurance').size()[0]/dataframe[dataframe['homeowner']==0].shape[0])
prob_In1_H0 = (dataframe[dataframe['homeowner']==0].groupby('insurance').size()[1]/dataframe[dataframe['homeowner']==0].shape[0])
odds2 = prob_In0_H0/prob_In1_H0

oddsRatio = odds1/odds2
print(oddsRatio)

######################################################################################
#Question3
print("---------------------------Question3----------------------------------------")
import pandas as pd
import numpy as np
import scipy.stats
import warnings
from sklearn import naive_bayes

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

def RowWithColumn (rowVar, columnVar, show = 'ROW'):   

    countTable = pd.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
    print("Frequency Table: \n", countTable)
    print()

    return

dataset = pd.read_csv(r'C:/Users/Akash1313/Desktop/CS584_ML/Assignment4/Purchase_Likelihood.csv', delimiter=',')

#Que 3a
print("-------------------------ans 3a-------------------------------")
cTable = pd.crosstab(index = dataset['insurance'], columns = ["Count"], margins = True, dropna = False)
cTable['Class Prob'] = cTable['Count'] / len(dataset)
cTable = cTable.drop(columns = ['All'])
print(cTable)

#Que 3b
print("-------------------------ans 3b-------------------------------")
RowWithColumn(dataset['insurance'], dataset['group_size'])

#Que 3c
print("-------------------------ans 3c-------------------------------")
RowWithColumn(dataset['insurance'], dataset['homeowner'])

#Que 3d
print("-------------------------ans 3d-------------------------------")
RowWithColumn(dataset['insurance'], dataset['married_couple'])

#Que 3e
print("-------------------------ans 3e-------------------------------")

def ChiSquareTest (xCat, yCat, debug = 'N'):
    
    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = np.sum(rTotal)
    expCount = np.outer(cTotal, (rTotal / nTotal))

    if (debug == 'Y'):
        print('Observed Count:\n', obsCount)
        print('Column Total:\n', cTotal)
        print('Row Total:\n', rTotal)
        print('Overall Total:\n', nTotal)
        print('Expected Count:\n', expCount)
        print('\n')
       
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()
    chiSqDf = (obsCount.shape[0] - 1.0) * (obsCount.shape[1] - 1.0)
    chiSqSig = scipy.stats.chi2.sf(chiSqStat, chiSqDf)

    cramerV = chiSqStat / nTotal
    if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = np.sqrt(cramerV)

    return(chiSqStat, chiSqDf, chiSqSig, cramerV)
     
catPred = ['group_size', 'homeowner', 'married_couple']
testResult = pd.DataFrame(index = catPred, columns = ['Statistic', 'DF', 'Significance', 'Association', 'Measure'])

for cp in catPred:
    chiSqStat, chiSqDf, chiSqSig, cramerV = ChiSquareTest(dataset[cp], dataset['insurance'], debug = 'Y')
    testResult.loc[cp] = [chiSqStat, chiSqDf, chiSqSig, 'Cramer''V', cramerV]

rankIn = testResult.sort_values('Measure', axis = 0, ascending = False)
print(rankIn)

#Que 3f 
print("-------------------------ans 3f-------------------------------")
trainX = dataset[catPred].astype('category')
trainY = dataset['insurance'].astype('category')

_objNB = naive_bayes.MultinomialNB(alpha = 1.0e-10)
thisFit = _objNB.fit(trainX, trainY)

test_data = []
for i in range(1, 5):
    for j in range(2):
        for k in range(2):
            test_data.append([i, j, k])

testX = pd.DataFrame(test_data, columns=['group_size','homeowner','married_couple'])
testX = testX[catPred].astype('category')
pred_prob = pd.DataFrame(_objNB.predict_proba(testX), columns = ['prob_I0', 'prob_I1','prob_I2'])
data = pd.concat([testX, pred_prob], axis = 1)
print(data)

print("-------------------------ans 3g-------------------------------")
data['oddVal(prob_I1/prob_I0)'] = data['prob_I1'] / data['prob_I0']
print(data[['group_size','homeowner','married_couple','oddVal(prob_I1/prob_I0)']])
max_row = data.loc[data['oddVal(prob_I1/prob_I0)'].idxmax()]
print("The maximum odd value is obtained when \ngroup_size =",max_row['group_size'],", homeowner = ",max_row['homeowner'],", married_couple = ",max_row['married_couple'])
print('The maximum odd value is: ',max_row['oddVal(prob_I1/prob_I0)'])     
