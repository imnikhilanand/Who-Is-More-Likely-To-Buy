# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 02:17:11 2022

@author: Nikhil
"""

""" Importing the libraries and files """

# importing the libraries
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
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
sns.barplot(x = "treatment_group_key", y="conversion" ,data=temp_distribution, palette='Blues')

# difference in treatment and control groups 
data.groupby('treatment_group_key').aggregate({'conversion':'mean'})['conversion'][1] - data.groupby('treatment_group_key').aggregate({'conversion':'mean'})['conversion'][0]

# checking the difference between the means of the two groups
control = data.query('treatment_group_key == 0 and conversion == 1')['conversion'].count()
treatment = data.query('treatment_group_key == 1 and conversion == 1')['conversion'].count()

# proportion z-test of the two groups
proportions_ztest([control, treatment], [5000, 5000])


""" Preparing the data for Modeling """

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



""" First Model S-Learner """
    
# building the model
model = XGBClassifier()

# setting the parameters
parameters = {'eta':[0.01, 0.1, 1],
              'max_depth':[1, 5, 9],
              'alpha':[0.1, 1],
              'lambda':[0.01, 0.1],
              'gamma':[1, 10]
              }

# grid search cv to find the best hyperparameters
clf = GridSearchCV(model, parameters)

# training the grid search model
clf.fit(X_train, y_train)    

# best parameters
clf.best_params_

# building the optimal model
model_xgb = XGBClassifier(eta=0.1,
                          max_depth=5,
                          alpha=1,
                          gamma=1)

# fitting the model using K-Fold cross validation
score  = cross_val_score(model_xgb, X_train, y_train, cv=5, scoring='accuracy')

# modeling again
model_xgb.fit(X_train, y_train)

# testing the dataset
y_pred = model_xgb.predict(X_test)

y_pred_prob = model_xgb.predict_proba(X_test)[::,1]

print(roc_auc_score(y_test,y_pred))

# plotting the AUC-ROC curve
fpr, tpr, _ = roc_curve(y_test,  y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


""" Second Model T-Learner """




