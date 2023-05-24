import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Read dataset
df=pd.read_csv('iris.csv')
print('Iris dataset is succeessfuly loaded....')
#Display information of dataset
print('Information of dataset:\n', df.info)
print('shape of dataset (row x column): ', df.shape)
print('Columns Name : ', df. columns)
print('Total element in dataset : ',df.size)
print('Datatype attributes(columns)', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)
#Find missing values
print('Missing values')
print(df.isnull().sum())
#Histogram of 1-variable
fig, axes = plt.subplots(2,2)
fig.suptitle('Histogram of 1-variable')
sns.histplot(data = df, x='sepal.length',ax=axes[0,0])
sns.histplot(data = df, x='sepal.length',ax=axes[0,1])
sns.histplot(data = df, x='sepal.length',ax=axes[1,0])
sns.histplot(data = df, x='sepal.length',ax=axes[1,1])
plt.show()
#Histogram of 2-variable
fig, axes = plt.subplots(2,2)
fig.suptitle('Histogram of 2-variable')
sns.histplot(data = df, x='sepal.length',hue='variety',multiple='dodge',ax=axes[0,0])
sns.histplot(data = df, x='sepal.length',hue='variety',multiple='dodge',ax=axes[0,1])
sns.histplot(data = df, x='sepal.length',hue='variety',multiple='dodge',ax=axes[1,0])
sns.histplot(data = df, x='sepal.length',hue='variety',multiple='dodge',ax=axes[1,1])
#Boxplot of 1-variable
fig, axes=plt.subplots(2,2)
fig.suptitle('Boxplot of 1-variable')
sns.histplot(data = df, x='sepal.length',ax=axes[0,0])
sns.histplot(data = df, x='sepal.length',ax=axes[0,1])
sns.histplot(data = df, x='sepal.length',ax=axes[1,0])
sns.histplot(data = df, x='sepal.length',ax=axes[1,1])
plt.show()
#Boxplot of 2-variable
fig, axes = plt.subplots(2,2)
fig.suptitle('Boxplot of 2-variable')
sns.histplot(data = df, x='sepal.length',hue='variety',multiple='dodge',ax=axes[0,0])
sns.histplot(data = df, x='sepal.length',hue='variety',multiple='dodge',ax=axes[0,1])
sns.histplot(data = df, x='sepal.length',hue='variety',multiple='dodge',ax=axes[1,0])
sns.histplot(data = df, x='sepal.length',hue='variety',multiple='dodge',ax=axes[1,1])
plt.show()

