'''
Created on 18 nov. 2015

@author: QuetzalGoute
'''

# Projet Kaggle Homesite

 
# Imports
from scipy import *
from numpy import *
from sklearn import *
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
from sklearn import preprocessing
import pandas
from pandas import *
from matplotlib import *
import matplotlib.pyplot as plt
import pylab


##############################################################
# 1 - Prepare the data
##############################################################

# Create the training & test sets
trainDataSetTemp=pandas.read_csv("Data/train.csv/train.csv", quotechar='"', skipinitialspace=True,iterator=True, chunksize=1000)
trainDataSet=concat(trainDataSetTemp, ignore_index=True)
testDataSetTemp=pandas.read_csv("Data/test.csv/testExtrait3.csv", quotechar='"', skipinitialspace=True,iterator=True, chunksize=200)
testDataSet=concat(testDataSetTemp, ignore_index=True)
                    
print('Taille de train et test (in bytes):')
print(sys.getsizeof(trainDataSet))
print(sys.getsizeof(testDataSet))

# Make a copy of the training and test data sets
train=trainDataSet.copy()
test=testDataSet.copy()

# ------------------------------------------------------------------------------------
# 1.1 - Data Visualisation 
#-------------------------------------------------------------------------------------

# Displaying info
stars="\n *****************************************************************\n"
print(stars,"Ligne 12 du trainDataSet:")
print(train.iloc[[12]])
#print(stars,"TrainDataSet:")
#print(trainDataSet)
print(stars, "Columns: \n")
print(train.columns)
print(stars,"Information - train: \n")
print(train.info())
print(stars, "Statistics and info - train: \n")
print(train.describe()) # dataFrame containing statistics and the same than info() options: include, exclude : list-like, all, or None (default))

# Histogramme de Field8 dans train ac matplotlib.pyplot
listField8=pandas.Series.tolist(train.Field8) # convert the pandas series to a list
plt.hist(listField8,20,normed=1)
plt.title('Field8 distribution (in train)')
plt.xlabel('Field8')
plt.ylabel('Nb occurrences')
plt.show()

# Histogramme de QuoteConversionFlag avec pylab
train.QuoteConversion_Flag.hist() # bins=2 si on veut en parametre et train.QuoteConversion_Flag.dropna().hist() pour retirer les valeurs manquantes
pylab.title("QuoteConversionFlag distribution (in train)")
pylab.ylabel("Nombre d'occurrences")
pylab.xlabel("QuoteConversionFlag")
pylab.show()

# Histogramme de Field8 dans train en indiquant la conversion
pandas.crosstab(train.Field8,train.QuoteConversion_Flag.astype(bool)).plot(kind='bar',stacked=True)
train.Field8.hist()
plt.title("Field8 distribution (in train)")
plt.ylabel("Nombre d'occurrences")
plt.xlabel("Field8")
plt.show()

# Histogramme de PropertyField31 dans train en indiquant la conversion
pandas.crosstab(train.PropertyField31,train.QuoteConversion_Flag.astype(bool)).plot(kind='bar',stacked=True)
plt.title("PropertyField31 distribution (in train)")
plt.ylabel("Nombre d'occurrences")
plt.xlabel("PropertyField31")
plt.show()

# Grouper les lignes par valeur de la cible
print(stars,'train - Means des colonnes suivant la valeur de QuoteconversionFlag')
print(train.groupby('QuoteConversion_Flag').mean())

# -------------------------------------------------------------------------------------
# 1.2 - Data Cleaning  - Traitements communs a toutes les methodes
#--------------------------------------------------------------------------------------

# Drop unnecessary columns (these columns won't be useful in analysis and prediction)
train = train.drop(['QuoteNumber'], axis=1)
test = test.drop(['QuoteNumber'], axis=1)

# Drop columns with many missing values (a half)
train = train.drop(['PersonalField84'], axis=1) # looks like only contains missing values and 2
test = test.drop(['PersonalField84'], axis=1)
train = train.drop(['PropertyField29'], axis=1) # looks like only contains missing values and 0
test = test.drop(['PropertyField29'], axis=1)

# Remove constant columns
#     PropertyField6, PropertyField9, et PropertyField20 n'ont que des 0 dans le trainExtrait3
#     PropertyField5 n'a que des Y dans le trainExtrait3
#     PersonalField64, PersonalField 65, PersonalField66, PersonalField67, 69 70 71, PersonalField72 n'a que des 0 dans le trainExtrait3
#     PersonalField68 et 73 n'a que des 1 dans le trainExtrait3
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

# Afficher les types des colonnes et les colonnes dont le type est object
print(stars,'Types des colonnes:')
print(train.dtypes)
print(stars,'Colonnes dont le type est object:')
print(train.dtypes[train.dtypes.map(lambda x: x=='object')])

# Drop rows with missing values
print(stars,'Dimensions de test:')
print(test.shape)
print(stars,'Dropping rows with missing values in test and train')
train=train.dropna()
test=test.dropna()
print(stars,"Information - train:\n")
print(train.info())
print(stars,'Dimensions de test:')
print(test.shape)


################################################################
# 2 - apply Machine Learning algorithms
################################################################

# --------------------------------------------------------------
# 2.1 Random Forests
# --------------------------------------------------------------

train1=train.copy()
test1=test.copy()

# Transforme une variable nominale en plusieurs var binaires
dummy_df=pandas.get_dummies(train1['Field12'], prefix='Field12_dummy')
train1=train1.join(dummy_df) #dummy_Field12
print(stars,"Field12 deviens dummy, train1:")
print(train1.info())
print(train1.columns)
dummy_dft=pandas.get_dummies(test1['Field12'], prefix='Field12_dummy')
test1=test1.join(dummy_dft)
print(stars,"Field12 deviens dummy, test1:")
print(test1.info())
 
dummy_df2=pandas.get_dummies(train1['CoverageField8'], prefix='CoverageField8_dummy')
train1=train1.join(dummy_df2)
dummy_df2t=pandas.get_dummies(test1['CoverageField8'], prefix='CoverageField8_dummy')
test1=test1.join(dummy_df2t)
 
dummy_df=pandas.get_dummies(train1['PersonalField7'], prefix='PersonalField7_dummy')
train1=train1.join(dummy_df)
dummy_df=pandas.get_dummies(test1['PersonalField7'], prefix='PersonalField7_dummy')
test1=test1.join(dummy_df)

dummy_df=pandas.get_dummies(train1['PropertyField3'], prefix='PropertyField3_dummy')
train1=train1.join(dummy_df)
dummy_df=pandas.get_dummies(test1['PropertyField3'], prefix='PropertyField3_dummy')
test1=test1.join(dummy_df)
 
dummy_df=pandas.get_dummies(train1['PropertyField4'], prefix='PropertyField4_dummy')
train1=train1.join(dummy_df)
dummy_df=pandas.get_dummies(test1['PropertyField4'], prefix='PropertyField4_dummy')
test1=test1.join(dummy_df)
 
dummy_df=pandas.get_dummies(train1['PropertyField28'], prefix='PropertyField28_dummy')
train1=train1.join(dummy_df)
dummy_df=pandas.get_dummies(test1['PropertyField28'], prefix='PropertyField28_dummy')
test1=test1.join(dummy_df)
 
dummy_df=pandas.get_dummies(train1['PropertyField30'], prefix='PropertyField30_dummy')
train1=train1.join(dummy_df)
dummy_df=pandas.get_dummies(test1['PropertyField30'], prefix='PropertyField30_dummy')
test1=test1.join(dummy_df)
 
dummy_df=pandas.get_dummies(train1['PropertyField32'], prefix='PropertyField32_dummy')
train1=train1.join(dummy_df)
dummy_df=pandas.get_dummies(test1['PropertyField32'], prefix='PropertyField32_dummy')
test1=test1.join(dummy_df)
# 
# dummy_df=pandas.get_dummies(train1['PropertyField34'], prefix='PropertyField34_dummy')
# train1=train1.join(dummy_df)
# dummy_df=pandas.get_dummies(test1['PropertyField34'], prefix='PropertyField34_dummy')
# test1=test1.join(dummy_df)
# 
# dummy_df=pandas.get_dummies(train1['PropertyField36'], prefix='PropertyField36_dummy')
# train1=train1.join(dummy_df)
# dummy_df=pandas.get_dummies(test1['PropertyField36'], prefix='PropertyField36_dummy')
# test1=test1.join(dummy_df)


# Prepare the train dataset
# RF only allow float values -> remove the strings columns
train1=train1.loc[:, train1.dtypes != object]
# ou train2 = train1.select_dtypes(exclude=[np.object]) 
print(stars, 'Remove object columns, new train1:')
print(train1.info())
# Convert train1 to a numpy array
train1 = train1.values
print(stars, 'Converting train1 to a numpy array, new array:')
print(train1)

# Prepare the test dataset in the same way
print(stars,'Dimensions de test1:')
print(test1.shape)
test1=test1.loc[:, test1.dtypes != object]
print(stars, 'Remove object columns, new test1:')
print(test1.info())
test1=test1.values

# Create the random forest object which will include all the parameters for the fit
nTrees=100
forest = RandomForestClassifier(n_estimators = nTrees,oob_score=True) #oob_score: Whether to use out-of-bag samples to estimate the generalization error.

# Fit the training data to the labels and create the decision trees
forest = forest.fit(train1[0::,1::],train1[0::,0])

# Score of the training dataset obtained using an out-of-bag estimate
print(stars,"Out of bag Score du Random Forest sur le training data set (Score of the training dataset obtained using an out-of-bag estimate):")
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

# ------------------------------------------------
# 2.2 Regression logistique
# ------------------------------------------------

train2=train.copy()
test2=test.copy()

print(stars,"Regression Logistique")

# Prepare the train dataset (The Logistic Regression only treats numeric values)
train2=train2.loc[:, train2.dtypes != object] # drop string columns
train2 = train2.values # Convert train1 to a numpy array

# Prepare the test dataset in the same way
test2=test2.loc[:, test2.dtypes != object]
test2=test2.values

# instantiate a logistic regression model, and fit with train
modelLogR = LogisticRegression()
modelLogR = modelLogR.fit(train2[0::,1::],train2[0::,0])

# check the accuracy on the training set
print(stars,'Check the accuracy on the training set')
print(modelLogR.score(train2[0::,1::],train2[0::,0]))

# predict class labels for the test set
predictedClassLogR = modelLogR.predict(test2)
print(stars,"Classes predites par la regression logistique (pour test):")
print(predictedClassLogR)

# generate class probabilities for the test set
probaLogR = modelLogR.predict_proba(test2)
print(stars,"Probabilites predites par la regression logistique (pour test):")
print(probaLogR)

# Do the prediction on the training set
predictedClassLogRtrain = modelLogR.predict(train2[0::,1::])
probaLogRtrain = modelLogR.predict_proba(train2[0::,1::])

# generate evaluation metrics
print(stars,"Evaluation metrics on train:")
print (metrics.accuracy_score(train2[0::,0], predictedClassLogRtrain))
print (metrics.roc_auc_score(train2[0::,0], probaLogRtrain[:, 1]))

# confusion matrix and a classification report
print(stars,"Confusion matrix and a classification report on train:")
print (metrics.confusion_matrix(train2[0::,0], predictedClassLogRtrain))
print (metrics.classification_report(train2[0::,0], predictedClassLogRtrain))

# Comparison of the Log Reg and the random forest
print(stars,"Comparison between the class predicted by the RF and the logistic Regression")
print(numpy.array(outputClassRf)-numpy.array(predictedClassLogR))

# -------------------------------------------------------------------
# 2.3 SVM
# -------------------------------------------------------------------

train3=train.copy()
test3=test.copy()

print(stars,"SVM")

# Prepare the train dataset (The Logistic Regression only treats numeric values)
train3=train3.loc[:, train3.dtypes != object] # drop string columns
train3 = train3.values # Convert train1 to a numpy array

# Prepare the test dataset in the same way
test3=test3.loc[:, test3.dtypes != object]
test3=test3.values

# Attention il faut rescaler les donnees pour le svm
#train3 = preprocessing.scale(train3)
#test3 = preprocessing.scale(test3)

# Construct an SVM classifier
svmClassifier = svm.SVC(probability=True)
svmClassifier.fit(train3[0::,1::],train3[0::,0])
predictedClassSvm = svmClassifier.predict(test3)
predictedProbaSvm = svmClassifier.predict_proba(test3)

# Comparison of the Log Reg and the random forest
print(stars,"Comparison between the class predicted by the SVM and the logistic Regression")
print(numpy.array(predictedClassSvm)-numpy.array(predictedClassLogR))

# Comparison of the Log Reg and the random forest
print(stars,"Comparison between the class predicted by the RF and SVM")
print(numpy.array(outputClassRf)-numpy.array(predictedClassSvm))



#------------------------------------------------------------------
# 3.4 Naive Bayes ou ann ou arbres renforces?
#----------------------------------------------------------------------




######################################################################
# 3- Make a submission file
######################################################################
#===============================================================================
# sampleSubmission = pandas.read_csv('Data/sample_submission.csv/sample_submission.csv')
# submission=sampleSubmission.copy()
# submission.QuoteConversion_Flag = pred_average
# submission.to_csv('Data/submission.csv', index=False)
#===============================================================================
