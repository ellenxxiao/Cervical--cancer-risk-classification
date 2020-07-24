#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:12:33 2020

@author: ellenxiao
"""

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# create another data set for feature selections. Drop correlated features based on above correlation heatmap
df2 = df.drop(['Dx','STDs','Smokes','IUD','Hormonal Contraceptives','STDs: Number of diagnosis',
               'Smokes (packs)','Hormonal Contraceptives (year/after first intercourse)'],axis=1)
# drop not quite understandable variables
df.drop(['Dx'],axis=1,inplace=True)
df2_target = df2['target']
df2_feature = df2.drop(['target'],axis=1)

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

''' RFE + L2-Regularized Logistic Regression '''
model = LogisticRegression('l2')
# create the RFE model and select 3 attributes
rfe = RFE(model, 10)
rfe = rfe.fit(df2_feature, df2_target)

result_lg = pd.DataFrame()
result_lg['Features'] = df2_feature.columns
result_lg ['Ranking'] = rfe.ranking_
result_lg.sort_values('Ranking', inplace=True ,ascending = False)

plt.figure(figsize=(10,10))
sns.set_color_codes("pastel")
sns.barplot(x = 'Ranking',y = 'Features', data=result_lg, color="tomato")
plt.show()

''' RandomForest Classification '''
rf2 = RandomForestClassifier()
rf2.fit(df2_feature,df2_target)
result_rf = pd.DataFrame()
result_rf['Features'] = df2_feature.columns
result_rf ['Importance'] = rf2.feature_importances_
result_rf.sort_values('Importance',inplace=True, ascending = False)

plt.figure(figsize=(11,11))
sns.set_color_codes("pastel")
sns.barplot(x = 'Importance',y = 'Features', data=result_rf, color="salmon")
plt.show()