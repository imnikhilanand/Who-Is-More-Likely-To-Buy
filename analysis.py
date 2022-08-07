# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 02:17:11 2022

@author: Nikhil
"""

""" importing the libraries and files """

# importing the libraries
import pandas as pd
import statsmodels

# importing the dataset
data = pd.read_csv('data/uplift_synthetic_data_100trials.csv', nrows=10000)

# encoding treatment variables
data['treatment_group_key'] = data['treatment_group_key'].apply(lambda x: 0 if x=='control' else 1)


""" Exploratory Data Analysis """

# columns in the dataframe
data.columns

# removing irrelevant columns
del data['Unnamed: 0']

# data points in different gorups (control and treatment)
data.groupby('treatment_group_key').aggregate({'treatment_group_key':['count']})

# lets check the average customer conversion in both the groups
data.groupby('treatment_group_key').aggregate({'conversion':['mean']})

# difference in treatment and control groups 
data.groupby('treatment_group_key').aggregate({'conversion':'mean'})['conversion'][1] - data.groupby('treatment_group_key').aggregate({'conversion':'mean'})['conversion'][0]

# checking the difference between the means of the two groups
control = data.query('treatment_group_key == 0 and conversion == 1')['conversion'].count()
treatment = data.query('treatment_group_key == 1 and conversion == 1')['conversion'].count()

# proportion z-test of the two groups
statsmodels.stats.proportion.proportions_ztest([control, treatment], [5000, 5000])
















