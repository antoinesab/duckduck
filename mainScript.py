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
trainDataSet=pandas.read_csv("Data/train.csv/trainExtrait3.csv", quotechar='"', skipinitialspace=True)
testDataSet=pandas.read_csv("Data/test.csv/testExtrait2.csv", quotechar='"', skipinitialspace=True)

print(sys.getsizeof(trainDataSet))
print(sys.getsizeof(testDataSet))

# Some displaying..
stars="\n *****************************************************************\n"
print(stars,"Ligne 12 du trainDataSet:")
print(trainDataSet.iloc[[12]])
#print(stars,"TrainDataSet:")
#print(trainDataSet)
print(stars, "Columns: \n")
print(trainDataSet.columns)
print(stars,"Information - train: \n")
print(trainDataSet.info())
print(stars, "Statistics and info - train: \n")
print(trainDataSet.describe()) # dataFrame containing statistics and the same than info() options: include, exclude : list-like, all, or None (default)

# Make a copy of the training and test data sets
train=trainDataSet.copy()
test=testDataSet.copy()

# Drop unnecessary columns (these columns won't be useful in analysis and prediction)
train = train.drop(['QuoteNumber'], axis=1)
test = test.drop(['QuoteNumber'], axis=1)

# Drop columns with many missing values (a half)
train = train.drop(['PersonalField84'], axis=1) # looks like only contains missing values and 2
test = test.drop(['PersonalField84'], axis=1)
train = train.drop(['PropertyField29'], axis=1) # looks like only contains missing values and 0
test = test.drop(['PropertyField29'], axis=1)

# Remove constant columns
# PropertyField6, PropertyField9, et PropertyField20 n'ont que des 0 dans le trainExtrait3
# PropertyField5 n'a que des Y dans le trainExtrait3
# PersonalField64, PersonalField 65, PersonalField66, PersonalField67, 69 70 71, PersonalField72 n'a que des 0 dans le trainExtrait3
# PersonalField68 et 73 n'a que des 1 dans le trainExtrait3
print(stars, "PropertyField6:")
print(train.describe().PropertyField6)
print(stars,"Dropping constant columns")
train=train.drop(['PropertyField6', 'PropertyField9', 'PropertyField20', 'PropertyField5', 'PersonalField64', 'PersonalField65', 'PersonalField66', 'PersonalField67', 'PersonalField69', 'PersonalField70', 'PersonalField71', 'PersonalField72', 'PersonalField68', 'PersonalField73'], axis=1)
test = test.drop(['PropertyField6', 'PropertyField9', 'PropertyField20', 'PropertyField5', 'PersonalField64', 'PersonalField65', 'PersonalField66', 'PersonalField67', 'PersonalField69', 'PersonalField70', 'PersonalField71', 'PersonalField72', 'PersonalField68', 'PersonalField73'], axis=1)

print(stars,"After dropping some columns, train:\n")
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

# Histogramme de Field8 dans train ac matplotlib.pyplot
listField8=pandas.Series.tolist(train.Field8) # convert the pandas series to a list
plt.hist(listField8,20,normed=1)
plt.title('Field8 in train')
plt.show()

# Histogramme de QuoteConversionFlag avec pylab
train.QuoteConversion_Flag.hist() # bins=2 si on veut en parametre
pylab.title("Histogramme de QuoteConversionFlag")
pylab.ylabel("Nombre d'occurrences")
pylab.show()

# Idem en retirant les valeurs manquantes
train.QuoteConversion_Flag.dropna().hist()
pylab.title("Histogramme de QuoteConversionFlag en retirant les valeurs manquantes")
pylab.ylabel("Nombre d'occurrences")
pylab.show()

# Afficher les types des colonnes
print(stars,'Types des colonnes:')
print(train.dtypes)

# Afficher les colonnes dont le type est object
print(stars,'Colonnes dont le type est object:')
print(train.dtypes[train.dtypes.map(lambda x: x=='object')])

print(stars,'Dimensions de test:')
print(test.shape)

# Drop rows with missing values
print(stars,'Dropping rows with missing values in test and train')
train=train.dropna()
test=test.dropna()
print(stars,"Inforamtion - train:\n")
print(train.info())

# Histogramme de QuoteConversionFlag avec pylab
train.QuoteConversion_Flag.hist() 
pylab.title("Histogramme de QuoteConversionFlag apres dropping des rows with missing values dans tout le train")
pylab.ylabel("Nombre d'occurrences")
pylab.show()

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
forest = RandomForestClassifier(n_estimators = nTrees,oob_score=True)

# Fit the training data to the labels and create the decision trees
forest = forest.fit(train1[0::,1::],train1[0::,0])

# Score of the training dataset obtained using an out-of-bag estimate
print(stars,"Score du Random Forest sur le training data set:")
print(forest.oob_score_)
print(stars,"Nombre de classes pour la sortie:")
print(forest.n_classes_)
print(stars,"Classes pour la sortie:")
print(forest.classes_)

# Take the same decision trees and run it on the test data
outputClassRf = forest.predict(test1)
print(stars,"Classes predites par le Random Forest:")
print(outputClassRf)
outputProbaRf = forest.predict_proba(test1)
print(stars,"Probabilites predites par le Random Forest:")
print(outputProbaRf)

# Score sur le train1
trainScoreRf=forest.score(train1[0::,1::],train1[0::,0])
print(stars, 'Score sur le train1:')
print(trainScoreRf)




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

