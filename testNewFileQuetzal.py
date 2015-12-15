'''
Created on 18 nov. 2015

@author: QuetzalRhead
'''

#############################
# Projet Kaggle Homesite
##############################
 
# Imports
from scipy import *
from numpy import *
from sklearn import *
import pandas

# Create the training & test sets
trainDataSet=pandas.read_csv("Data/train.csv/trainExtrait.csv", quotechar='"', skipinitialspace=True)

# Some displaying..
print(trainDataSet)
print(trainDataSet.columns)
print(trainDataSet.describe()) # dataFrame containing statistics on the values of each column

# Make a copy of the training data set
train=trainDataSet.copy()
