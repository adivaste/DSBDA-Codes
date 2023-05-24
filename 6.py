import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def removeout(df,var):

	Q1=df[var].quantile(0.25)
	Q3=df[var].quantile(0.75)

	IQR=Q3-Q1

	low=Q1-1.5*IQR
	high=Q3+1.5*IQR

	df=df[(df[val]>=low) & (df[val]<=high)]
	return df

df=pd.read_csv('iris.csv')
print("Dataset loaded succesfully.....")


choice=1

while(choice!=10):
	print("--------------------Menu--------------")
	print("1.Information Of Dataset")
	print("2.Statistical Information of Dataset")
	print("3.Shape of Dataset")
	print("4.Find Missing values")
	print("5.Detect outlier and remove it")
	print("6.correlation matrix")
	print("7.Label Encoding of variable variety")
	choice=(int(input("Enter Your Choice : ")))
	if(choice==1):
		print("Information of Dataset \n",df.info())
		print("First 5 rows \n",df.head().T)
		print("Last 5 rows \n",df.tail().T)

	if(choice==2):
		print("Statistical Information \n",df.describe())
	if(choice==3):
		print("shape of Dataset \n",df.shape)

	if(choice==4):
		print("Missing Values count ", df.isna().count())
		print("Missing values sum " , df.isna().sum())
	if(choice==5):

		new=['sepal.length','sepal.width','petal.length','petal.width']

		fig,axes=plt.subplots(2,2)
		fig.suptitle("Before removing outlier")
		
		sns.boxplot(data=df,x='sepal.length',ax=axes[0,0])
		sns.boxplot(data=df,x='sepal.width',ax=axes[0,1])
		sns.boxplot(data=df,x='petal.length',ax=axes[1,0])
		sns.boxplot(data=df,x='petal.width',ax=axes[1,1])
		fig.tight_layout()
		plt.show()
	if(choice==6):
		print(df.corr())
		sns.heatmap(df.corr(),annot=True)
		plt.show()
	if(choice==7):
		df['variety']=df['variety'].astype('category')
		df['variety']=df['variety'].cat.codes
	if(choice==8):

		x=df.iloc[:,[0,1,2,3]].values
		y=df.iloc[:,4].values

		# x=df[['sepal.length','sepal.width','petal.length','petal.width']]
		# y=df['variety']
		from sklearn.model_selection import train_test_split

		xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=11)

		from sklearn.preprocessing import StandardScaler

		xtrain=StandardScaler().fit_transform(xtrain)
		xtest=StandardScaler().fit_transform(xtest)

		from sklearn.naive_bayes import GaussianNB
		gb=GaussianNB()
		gb.fit(xtrain,ytrain)

		ypredict=gb.predict(xtest)
		print("Ypredicted \n",ypredict)

		sco=gb.score(xtest,ytest)
		print("Xtest \n" ,xtest)
		print("Ytest \n",ytest)
		print("Score>>>>>",sco)

	if(choice==9):

		from sklearn.metrics import confusion_matrix,classification_report

		cm=confusion_matrix(ytest,ypredict)
		print(cm)
		sns.heatmap(cm,annot=True)
		plt.show()
        
		print(classification_report(ytest,ypredict))

