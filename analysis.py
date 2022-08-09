# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 02:17:11 2022

@author: Nikhil
"""

""" Importing the libraries and files """

# importing the libraries
import pandas as pd
import statsmodels
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
sns.dark_palette("#69d", reverse=True, as_cmap=True)
sns.color_palette("Set2")

# importing the dataset
data = pd.read_csv('data/uplift_synthetic_data_100trials.csv', nrows=10000)

# encoding treatment variables
data['treatment_group_key'] = data['treatment_group_key'].apply(lambda x: 0 if x=='control' else 1)


""" Exploratory Data Analysis """

# columns in the dataframe
data.columns

# removing irrelevant columns
del data['Unnamed: 0']

# data points in different groups (control and treatment)
data.groupby('treatment_group_key').aggregate({'treatment_group_key':['count']})

# lets check the average customer conversion in both the groups
data.groupby('treatment_group_key').aggregate({'conversion':['mean']})

# plotting the distribution between two 
temp_distribution = data.groupby('treatment_group_key').aggregate({'conversion':'mean'}).reset_index()
sns.barplot(x = "treatment_group_key", y="conversion" ,data=temp_distribution, orient='h', palette='Blues')

# difference in treatment and control groups 
data.groupby('treatment_group_key').aggregate({'conversion':'mean'})['conversion'][1] - data.groupby('treatment_group_key').aggregate({'conversion':'mean'})['conversion'][0]

# checking the difference between the means of the two groups
control = data.query('treatment_group_key == 0 and conversion == 1')['conversion'].count()
treatment = data.query('treatment_group_key == 1 and conversion == 1')['conversion'].count()

# proportion z-test of the two groups
statsmodels.stats.proportion.proportions_ztest([control, treatment], [5000, 5000])


""" Modeling """

# seleccting features that are required to build the model 
data = data[['treatment_group_key', 
             'conversion',
             'x1_informative', 
             'x2_informative',
             'x3_informative',
             'x4_informative', 
             'x5_informative', 
             'x6_informative',
             'x7_informative', 
             'x8_informative', 
             'x9_informative', 
             'x10_informative',
             'x11_irrelevant', 
             'x12_irrelevant', 
             'x13_irrelevant', 
             'x14_irrelevant',
             'x15_irrelevant',
             'x16_irrelevant', 
             'x17_irrelevant', 
             'x18_irrelevant',
             'x19_irrelevant',
             'x20_irrelevant',
             'x21_irrelevant', 
             'x22_irrelevant',
             'x23_irrelevant', 
             'x24_irrelevant', 
             'x25_irrelevant', 
             'x26_irrelevant',
             'x27_irrelevant', 
             'x28_irrelevant', 
             'x29_irrelevant', 
             'x30_irrelevant',
             'x31_uplift_increase', 
             'x32_uplift_increase', 
             'x33_uplift_increase',
             'x34_uplift_increase', 
             'x35_uplift_increase', 
             'x36_uplift_increase'
             ]]

# getting the predictors and results
X = data[['treatment_group_key', 
          'x1_informative', 
          'x2_informative',
          'x3_informative',
          'x4_informative', 
          'x5_informative', 
          'x6_informative',
          'x7_informative', 
          'x8_informative', 
          'x9_informative', 
          'x10_informative',
          'x11_irrelevant', 
          'x12_irrelevant', 
          'x13_irrelevant', 
          'x14_irrelevant',
          'x15_irrelevant',
          'x16_irrelevant', 
          'x17_irrelevant', 
          'x18_irrelevant',
          'x19_irrelevant',
          'x20_irrelevant',
          'x21_irrelevant', 
          'x22_irrelevant',
          'x23_irrelevant', 
          'x24_irrelevant', 
          'x25_irrelevant', 
          'x26_irrelevant',
          'x27_irrelevant', 
          'x28_irrelevant', 
          'x29_irrelevant', 
          'x30_irrelevant',
          'x31_uplift_increase', 
          'x32_uplift_increase', 
          'x33_uplift_increase',
          'x34_uplift_increase', 
          'x35_uplift_increase', 
          'x36_uplift_increase']]

y = data['conversion']

# splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )

# building the model
model = XGBClassifier()

# fitting the model using K-Fold cross validation
score  = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

print(score)





