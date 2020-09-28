# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 00:16:12 2020

@author: Lucia
"""

import pandas as pd
import numpy as np
df = pd.read_csv('D:\Desktop\IE517 ML in Fin Lab\Module5\HW5\hw5_treasury yield curve data.csv')

X = df.iloc[:,1:31]
y = df.iloc[:,31]


# Part 1: Exploratory Data Analysis


# heatmap
import matplotlib.pyplot as plt
import seaborn as sns
col = ['SVENF05','SVENF10','SVENF15','SVENF20','SVENF25','SVENF30']
sns.heatmap(df[col].corr(), annot = True, annot_kws = {"size": 7})
plt.yticks(rotation = 0, size = 14)
plt.xticks(rotation = 90, size = 14)  # fix ticklabel directions and size
plt.tight_layout()
plt.show()

# scatterplot matrix
col = ['SVENF03','SVENF06','SVENF09','SVENF12','SVENF15','SVENF18','SVENF21','SVENF24','SVENF27','SVENF30']
sns.pairplot(df[col], height = 2.5)
plt.tight_layout()
plt.show()

# Box plot
sns.boxplot(data = df)
plt.xlabel('Attribute')
plt.ylabel('Quantile Ranges')
plt.title("Box plot")
plt.xticks(rotation = 90)
plt.show()

# Split data into training and test sets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=42)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# Part 2: Perform a PCA on the Treasury Yield dataset


# PCA for all components
from sklearn.decomposition import PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
pca.explained_variance_ratio_

# PCA for n_components=3
pca_3 = PCA(n_components=3)
X_train_pca_3 = pca_3.fit_transform(X_train_std)
X_test_pca_3 = pca_3.transform(X_test_std)
pca_3.explained_variance_ratio_

# cumulative explained variance of the 3 component version
pca_3.explained_variance_ratio_[0]+pca_3.explained_variance_ratio_[1]+pca_3.explained_variance_ratio_[2]


# Part 3: Linear regression v. SVM regressor - baseline


# Linear model for all features
from sklearn.linear_model import LinearRegression
lr_all = LinearRegression()
lr_all.fit(X_train_std, y_train)
y_train_pred_lr_all = lr_all.predict(X_train_std)
y_test_pred_lr_all = lr_all.predict(X_test_std)
print(lr_all.coef_)

# Calculate its accuracy R2 score and RMSE for both in sample and out of sample
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y_train, y_train_pred_lr_all)),
        sqrt(mean_squared_error(y_test, y_test_pred_lr_all))))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred_lr_all),
        r2_score(y_test, y_test_pred_lr_all)))

# Linear regression for PCA transformed data
lr_3 = LinearRegression()
lr_3.fit(X_train_pca_3, y_train)
y_train_pred_lr_3 = lr_3.predict(X_train_pca_3)
y_test_pred_lr_3 = lr_3.predict(X_test_pca_3)
print(lr_3.coef_)

# Calculate its accuracy R2 score and RMSE for both in sample and out of sample
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y_train, y_train_pred_lr_3)),
        sqrt(mean_squared_error(y_test, y_test_pred_lr_3))))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred_lr_3),
        r2_score(y_test, y_test_pred_lr_3)))

# SVM for all features
from sklearn import svm
svm_all = svm.SVR(kernel='poly')
svm_all.fit(X_train_std, y_train)
y_train_pred_svm_all = svm_all.predict(X_train_std)
y_test_pred_svm_all = svm_all.predict(X_test_std)

# Calculate its accuracy
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y_train, y_train_pred_svm_all)),
        sqrt(mean_squared_error(y_test, y_test_pred_svm_all))))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred_svm_all),
        r2_score(y_test, y_test_pred_svm_all)))

# SVM for PCA transformed data
svm_3 = svm.SVR(kernel='poly')
svm_3.fit(X_train_pca_3, y_train)
y_train_pred_svm_3 = svm_3.predict(X_train_pca_3)
y_test_pred_svm_3 = svm_3.predict(X_test_pca_3)

# Calculate its accuracy
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y_train, y_train_pred_svm_3)),
        sqrt(mean_squared_error(y_test, y_test_pred_svm_3))))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred_svm_3),
        r2_score(y_test, y_test_pred_svm_3)))



print("-----------")
print("My name is Lu Yuan")
print("My NetID is: luyuan3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")