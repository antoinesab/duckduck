'''
Created on 18 nov. 2015

@author: QuetzalGoute
'''

# Projet Kaggle Homesite
##############################################################
 
# Imports
from scipy import *
from numpy import *
from sklearn import *
from sklearn.ensemble import RandomForestClassifier 
import pandas
from pandas import *
from matplotlib import *
import matplotlib.pyplot as plt
import pylab


##############################################################
# 1 - Prepare the data
##############################################################

# Create the training & test sets
trainDataSet=pandas.read_csv("Data/train.csv/trainExtrait2.csv", quotechar='"', skipinitialspace=True)
testDataSet=pandas.read_csv("Data/test.csv/testExtrait2.csv", quotechar='"', skipinitialspace=True)

# Some displaying..
stars="\n *****************************************************************\n"
print(trainDataSet)
print(stars, "Columns: \n")
print(trainDataSet.columns)
print(stars,"Information: \n")
print(trainDataSet.info())
print(stars, "Statistics and info: \n")
print(trainDataSet.describe()) # dataFrame containing statistics and the same than info()

# Make a copy of the training and test data sets
train=trainDataSet.copy()
test=testDataSet.copy()

# Remove constant columns-> drop unnecessary columns, these columns won't be useful in analysis and prediction
train = train.drop(['QuoteNumber'], axis=1)
test = test.drop(['QuoteNumber'], axis=1)
print(stars,"After drop:\n")
print(train.info())

# Convert Date to Year, Month, and Week
train['Year']  = train['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
train['Week']  = train['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))
test['Year']  = test['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
test['Week']  = test['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))
train.drop(['Original_Quote_Date'], axis=1,inplace=True)
test.drop(['Original_Quote_Date'], axis=1,inplace=True)
print(stars,"Create columns for the Date\n :")
print(train.info())

# Essaie de faire un histogramme de Field8 dans train ac matplotlib.pyplot
listField8=pandas.Series.tolist(train.Field8) # convert the pandas series to a list
plt.hist(listField8,20,normed=1)
plt.title('Field8 in train')
plt.show()

# idem avec matplotlib.pylab
train.Field7.hist()
pylab.show()

# en version amelioree dropna pour sup les valeurs manquantes
train.Field7.dropna().hist(bins=16)
pylab.show()

# obtenir les types des colonnes
print(stars,'Types des colonnes:')
print(train.dtypes)

# afficher les coonnes dont le type est object
print(stars,'Colonnes dont le type est object:')
print(train.dtypes[train.dtypes.map(lambda x: x=='object')])

print(stars,'Dimensions de test:')
print(test.shape)

# Drop rows with missing values
print(stars,'Dropping rows with missing values in test and train')
train=train.dropna()
test=test.dropna()

print(stars,'Dimensions de test:')
print(test.shape)

################################################################
# 2 - apply Machine Learning algorithms
################################################################

# 2.1 Random Forests

# Prepare the train dataset
# RF only allow float values -> remove the strings columns
train1=train.copy()
train1=train1.loc[:, train1.dtypes != object]
# ou train2 = train1.select_dtypes(exclude=[np.object]) 
print(stars, 'Remove object columns, new train1:')
print(train1.info())
# Convert train1 to a numpy array
train1 = train1.values
print(stars, 'Converting train1 to a numpy array, new array:')
print(train1)

# Prepare the test dataset idem
test1=test.copy()
print(stars,'Dimensions de test1:')
print(test1.shape)
test1=test1.loc[:, test1.dtypes != object]
print(stars, 'Remove object columns, new test1:')
print(test1.info())
test1=test1.values

# Create the random forest object which will include all the parameters for the fit
nTrees=100
forest = RandomForestClassifier(n_estimators = nTrees)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train1[0::,1::],train1[0::,0])

# Take the same decision trees and run it on the test data
outputRf = forest.predict(test1)
print(stars,"Output of the Random forest:")
print(outputRf)




# 2.2 Regression logistique



######################################################################
# 3- Make a submission file
######################################################################
#===============================================================================
# sampleSubmission = pandas.read_csv('Data/sample_submission.csv/sample_submission.csv')
# submission=sampleSubmission.copy()
# submission.QuoteConversion_Flag = pred_average
# submission.to_csv('Data/submission.csv', index=False)
#===============================================================================
