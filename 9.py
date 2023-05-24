import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Read Dataset
df = pd.read_csv('titanic.csv')
print('Titanic dataset is successfully loaded .....\n')
#Display information of dataset
print('Information of dataset:\n', df.info)
print('shape of dataset (row x column): ', df.shape)
print('Columns Name : ', df. columns)
print('Total element in dataset : ',df.size)
print('Datatype attributes(columns)', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)

# Find missing values
print('Missing values')
print(df.isnull().sum())
#  Fill the missing values
df['Age'].fillna(df['Age'].median(),inplace=True)
print('Null values are: \n',df.isnull().sum())


#Boxplot of 1-variable
fig, axes = plt.subplots(1,2)
fig.suptitle('Boxplot of 1-variable (Age & Fare)')
sns.histplot(data = df, x='Age',ax=axes[0])
sns.histplot(data = df, x='Fare',ax=axes[1])
plt.show()
#Boxplot of 2-variable
fig, axes = plt.subplots(2,2)
fig.suptitle('Boxplot of 2-variable')
sns.histplot(data = df, x='Survived',y='Age',hue='Survived',ax=axes[0,0])
sns.histplot(data = df, x='Survived',y='Age',hue='Survived',ax=axes[0,1])
sns.histplot(data = df, x='Sex',y='Age',hue='Sex',ax=axes[1,0])
sns.histplot(data = df, x='Sex',y='Fare',hue='Sex',ax=axes[1,1])
plt.show()
# Boxplot of 3-variable
fig, axes = plt.subplots(1,2)
fig.suptitle('Boxplot of 3-variable')
sns.histplot(data = df, x='Sex',y='Age',ax=axes[0])
sns.histplot(data = df, x='Sex',y='Fare',ax=axes[1])
plt.show()
