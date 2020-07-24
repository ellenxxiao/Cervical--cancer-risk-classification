#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:11:06 2020

@author: ellenxiao
"""

# add_trace function
def add_bar(feature,row,col):
    
    sub_df = pd.DataFrame(df.groupby(by=['target',feature],as_index=False)['Dx'].count())
    fig.add_trace(go.Bar(x=sub_df.loc[sub_df['target']==0][feature],
                     y=sub_df.loc[sub_df['target']==0]['Dx'],
                     name='No Disease',marker_color='lightblue'),row=row, col=col)
    fig.add_trace(go.Bar(x=sub_df.loc[sub_df['target']==1][feature],
                     y=sub_df.loc[sub_df['target']==1]['Dx'],
                     name='Diseae',marker_color='lightpink'),row=row, col=col)
    
fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Smokes and Cervical Cancer',
                                    'Hormonal Contraceptives and Cervical Cancer',
                                    'IUD and Cervical Cancer',
                                    'STDs and Cervical Cancer'))

add_bar('Smokes',1,1)
add_bar('Hormonal Contraceptives',1,2)
add_bar('IUD',2,1)
add_bar('STDs',2,2)

fig.update_layout(height=800, width=900, title_text="")
fig.show(renderer='svg')

partner_df = pd.DataFrame(df.groupby(by=['target','Number of sexual partners'],as_index=False)['Dx'].count())
partner_df_y = partner_df.loc[partner_df['target']==1]

preg_df = pd.DataFrame(df.groupby(by=['target','Num of pregnancies'],as_index=False)['Dx'].count())
preg_df_y = preg_df.loc[preg_df['target']==1]

# add_trace function
def add_bar2(feature,row,col,color):
    sub_df = pd.DataFrame(df.groupby(by=['target',feature],as_index=False)['Dx'].count())
    sub_df_y = sub_df.loc[sub_df['target']==1]
    fig.add_trace(go.Bar(x=sub_df_y.loc[sub_df_y['target']==1][feature],
                     y=sub_df_y.loc[sub_df_y['target']==1]['Dx'],
                     name='Disease',marker_color=color),row=row,col=col)
    
fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=('Number of sexual partners',
                                   'Number of pregnancies',
                                   'Smokes (years)',
                                   'Smokes (pack/year)',
                                   'Hormonal Contraceptives (years)',
                                   'IUD (years)'))
add_bar2('Number of sexual partners',1,1,'gold')
add_bar2('Num of pregnancies',1,2,'plum')
add_bar2('Smokes (years)',2,1,'lightsalmon')
add_bar2('Smokes (packs/year)',2,2,'thistle')
add_bar2('Hormonal Contraceptives (years)',3,1,'lightcoral')
add_bar2('IUD (years)',3,2,'lightpink')
fig.update_layout(height=700, width=700, title_text="")
fig.show(renderer='svg')

# add_trace function
def add_bar2(feature,row,col,color):
    sub_df = pd.DataFrame(df.groupby(by=['target',feature],as_index=False)['Dx'].count())
    sub_df_y = sub_df.loc[(sub_df['target']==1) & (sub_df[feature])]
    fig.add_trace(go.Bar(x=sub_df_y.loc[sub_df_y['target']==1][feature],
                     y=sub_df_y.loc[sub_df_y['target']==1]['Dx'],
                     name='Disease',marker_color=color),row=row,col=col)
fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=('Number of sexual partners',
                                   'Number of pregnancies',
                                   'Smokes (years)',
                                   'Smokes (pack/year)',
                                   'Hormonal Contraceptives (years)',
                                   'IUD (years)'))
add_bar2('Number of sexual partners',1,1,'gold')
add_bar2('Num of pregnancies',1,2,'plum')
add_bar2('Smokes (years)',2,1,'lightsalmon')
add_bar2('Smokes (packs/year)',2,2,'thistle')
add_bar2('Hormonal Contraceptives (years)',3,1,'lightcoral')
add_bar2('IUD (years)',3,2,'lightpink')
fig.update_layout(height=700, width=700, title_text="")
fig.show(renderer='svg')

cancer_df = df.loc[df['target']==1]
cancer_df.head()
# add_trace function
def add_hist(feature,row,col,color):
    fig.add_trace(go.Histogram(x=cancer_df[feature],
                     marker_color=color),row=row,col=col)
fig = make_subplots(
                    rows=2, cols=3,
                    subplot_titles=('Age',
                                    'First sexual intercourse',
                                    'Smokes (years)',
                                   'Smokes (pack/year)',
                                   'Hormonal Contraceptives (years)',
                                   'IUD (years)'))
add_hist('Age',1,1,'lightskyblue')
add_hist('First sexual intercourse',1,2,'thistle')
add_hist('Smokes (years)',1,3,'dodgerblue')
add_hist('Smokes (packs/year)',2,1,'orchid')
add_hist('Hormonal Contraceptives (years)',2,2,'mediumturquoise')
add_hist('IUD (years)',2,3,'mediumorchid')
fig.update_layout(height=700, width=1000, title_text="")
fig.show(renderer='svg')

''' Feature correlation'''
# Feature correlations within them
df_feature = df.drop(['target'],axis=1)
df_corr1 = df_feature.corr()
corr = go.Heatmap(x=df_corr1.columns,y=df_corr1.columns,z=df_corr1,type='heatmap',colorscale='Viridis')
data = [corr]
fig = go.Figure(data=data)
fig.show()
# Feature correlation between features and target
corrmat = df.corr()
k = 25 #number of variables for heatmap
cols = corrmat.nlargest(k,'target')['target'].index

cm = df[cols].corr()

plt.figure(figsize=(12,12))

sns.set(font_scale=1.25)
hm = sns.heatmap(cm,cmap = 'Set2', cbar=True, annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels = cols.values, xticklabels = cols.values)
plt.show()
