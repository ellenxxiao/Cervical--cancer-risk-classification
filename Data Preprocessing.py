#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:05:27 2020

@author: ellenxiao
"""

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly as ply
import seaborn as sns
from sklearn.model_selection import train_test_split
import plotly.io as pio

pio.renderers

file_name = "/Users/risk_factors_cervical_cancer.csv"
df = pd.read_csv(file_name, encoding="ISO-8859-1")
df.head()

print('Number of Rows: ',df.shape[0])
print('Number of Columns: ', df.shape[1])
print('Features: \n', df.columns.tolist())
print('Unique values: \n', df.nunique())

''' Fill/remove missing values'''
df[df=='?'].count()

df.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'],inplace=True,axis=1)
# fill features that have less than 100 NaN values with median values
df = df.replace('?',np.NaN)
df = df.apply(pd.to_numeric,axis=0) #convert to numerical

def fill_val(feature):
    df[feature].fillna(df[feature].median(),inplace=True)
fill_val('First sexual intercourse')
fill_val('Num of pregnancies')
fill_val('Number of sexual partners')

def smoke(feature):
    if feature == 'Smokes':
        df[feature].fillna(0,inplace=True)
    else:
        smokes = (df['Smokes']==1)
        df.loc[smokes,feature] = df.loc[smokes,feature].fillna(df.loc[smokes,feature].median())
        nosmokes = (df['Smokes']==0)
        df.loc[nosmokes,feature] = df.loc[nosmokes,feature].fillna(0)
smoke('Smokes')
smoke('Smokes (years)')
smoke('Smokes (packs/year)')

# Hormonal Contraceptive - fill NaN values with correlated features
df_data = df.drop(['Hinselmann','Schiller','Citology','Biopsy'], axis = 1)
df_corr = df_data.drop(['STDs:condylomatosis', 'STDs:cervical condylomatosis', 
                   'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 
                   'STDs:pelvic inflammatory disease', 'STDs:genital herpes', 'STDs:molluscum contagiosum', 
                   'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV'],axis=1)
df_corr = df_corr.dropna()
df_corr = df_corr.corr()

corr = go.Heatmap(x=df_corr.columns,y=df_corr.columns,z=df_corr,type='heatmap',colorscale='Viridis')
data = [corr]
fig = go.Figure(data=data)
fig.show(renderer='svg')

# From the correlation heatmap above, taking HC has a slightly relationship with the number of pregnancies.
preg = (df['Num of pregnancies']<df['Num of pregnancies'].mean())
df.loc[preg,'Hormonal Contraceptives'] = df.loc[preg,'Hormonal Contraceptives'].fillna(1)
df['Hormonal Contraceptives'].fillna(0,inplace=True)
hc = (df['Hormonal Contraceptives']==1)
df.loc[hc,'Hormonal Contraceptives (years)'] = df.loc[hc,'Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].median())
hc = (df['Hormonal Contraceptives']==0)
df.loc[hc,'Hormonal Contraceptives (years)'] = df.loc[hc,'Hormonal Contraceptives (years)'].fillna(0)

# The heatmap shows that age and number of pregnancies have relationships with IUD
age = (df['Age']>df['Age'].mean())
df.loc[age,'IUD'] = df.loc[age,'IUD'].fillna(1)
df.loc[preg,'IUD'] = df.loc[preg,'IUD'].fillna(1)
df['IUD'].fillna(0,inplace=True)
iud = (df['IUD']==1)
df.loc[iud,'IUD (years)'] = df.loc[iud,'IUD (years)'].fillna(df['IUD (years)'].median())
iud = (df['IUD']==0)
df.loc[iud,'IUD (years)'] = df.loc[iud,'IUD (years)'].fillna(0)

df.dropna(inplace=True)
df.drop(['Dx:Cancer','Dx:CIN','Dx:HPV'],axis=1,inplace=True)

# set our target 
df['target'] = df.apply(lambda row: 1 if (row['Hinselmann']+row['Schiller']+row['Citology']+row['Biopsy'])>=1 else 0, axis=1)
df_copy = df.copy()
df.drop(['Hinselmann','Schiller','Citology','Biopsy'],axis=1,inplace=True)
df.head()


''' Create features'''
# percentage of smoke year
df['Smokes (year/age)'] = round(df['Smokes (years)']/df['Age'],3)
# percentage Hormonal Contraceptive of years after first sexual intercourse
df['Hormonal Contraceptives (year/after first intercourse)'] = round(df['Hormonal Contraceptives (years)']/(df['Age']-df['First sexual intercourse']),3)
# percentage IUD of years after first sexual intercourse
df['IUD (year/after first intercourse)'] = round(df['IUD (years)']/(df['Age']-df['First sexual intercourse']),3)
# total packs smokes
df['Smokes (packs)'] = df['Smokes (packs/year)']*df['Smokes (years)']

pd.set_option('use_inf_as_na', True)

# Because we have Age = First Sexual Intercourse
df['Hormonal Contraceptives (year/after first intercourse)']=df['Hormonal Contraceptives (year/after first intercourse)'].fillna(0.000)
df['IUD (year/after first intercourse)']=df['IUD (year/after first intercourse)'].fillna(0.000)

# convert float to int
for col in df.columns:
    if col not in ['Smokes (year/age)','Hormonal Contraceptives (year/after first intercourse)','IUD (year/after first intercourse)']:
        df[col] = df[col].astype(int)
df.head()

