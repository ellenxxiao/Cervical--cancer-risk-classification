#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:14:53 2020

@author: ellenxiao
"""

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# Split train and test dataset
df_target = df['target']
X_train,X_test,y_train,y_test = train_test_split(df_feature,df_target,test_size=0.3)
df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)
df_y_train = pd.DataFrame(y_train)
df_y_test = pd.DataFrame(y_test)

# Categorical variables and numerical variables
num_val = ['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes (years)', 
           'Smokes (packs/year)', 'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',
           'STDs: Number of diagnosis', 'Smokes (year/age)','Hormonal Contraceptives (year/after first intercourse)',
           'IUD (year/after first intercourse)', 'Smokes (packs)']
cal_val = [col for col in df_feature if col not in num_val]
print(cal_val)

# Standardize data
scaler = StandardScaler()
df_train[num_val] = scaler.fit_transform(df_train[num_val])


''' L2-Regularized Logistic Regression'''
# performance of Logistic Regression
# Standardize data
scaler = StandardScaler()
df_test[num_val] = scaler.fit_transform(df_test[num_val])

model = LogisticRegression('l2')
model.fit(df_test,df_y_test)
prediction = model.predict(df_test)
score = model.score(df_test, df_y_test)
print('Accuracy: ', score)
cm = metrics.confusion_matrix(df_y_test, prediction)
f1_score = metrics.f1_score(df_y_test,prediction)
print('F1-Score:', f1_score)
recall = (cm[1][1])/(cm[1][0]+cm[1][1])
print('Sensitivity:', recall)

plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)

''' L10Regularized Logistic Regressiont'''
# performance of Logistic Regression
# Standardize data
scaler = StandardScaler()
df_test[num_val] = scaler.fit_transform(df_test[num_val])

model = LogisticRegression('l1')
model.fit(df_test,df_y_test)
prediction = model.predict(df_test)
score = model.score(df_test, df_y_test)
print('Accuracy: ', score)
cm = metrics.confusion_matrix(df_y_test, prediction)
f1_score = metrics.f1_score(df_y_test,prediction)
print('F1-Score:', f1_score)
recall = (cm[1][1])/(cm[1][0]+cm[1][1])
print('Sensitivity:', recall)

plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)

''' RandomForest Classification'''
# performance of Logistic Regression
# Standardize data
scaler = StandardScaler()
df_test[num_val] = scaler.fit_transform(df_test[num_val])

rf = RandomForestClassifier()
rf.fit(df_train, df_y_train)

rf.fit(df_test,df_y_test)
prediction = rf.predict(df_test)
score = rf.score(df_test, df_y_test)
print('Accuracy: ', score)
cm = metrics.confusion_matrix(df_y_test, prediction)
f1_score = metrics.f1_score(df_y_test,prediction)
print('F1-Score:', f1_score)
recall = (cm[1][1])/(cm[1][0]+cm[1][1])
print('Sensitivity:', recall)

plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)