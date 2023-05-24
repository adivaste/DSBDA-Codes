import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def removeout(df,var):
	Q1=df[var].quantile(0.25)
	Q3=df[var].quantile(0.75)

	IQR=Q3-Q1

	mini=Q1-1.5*IQR
	hig=Q3+1.5*IQR

	df=df[(df[var]>=mini) & (df[var]<=hig)]
	print("outliers removed...........")
	return(df)

df=pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')


choice=1

while(choice!=10):

	print("---------------menu-----------------")

	print("1.Information of dataset")
	print("2.Statistical values of data set")
	print("3.change data type")
	print("4.outliers remove")
	print("5.correlation")
	


	choice=int(input("Enter Your choice : "))

	if(choice==1):
		print("Informatio \n",df.info)
		print("Data Type \n",df.dtypes)
		print("Shape \n",df.shape)
		print("TOp 5 rows \n",df.head())
		print("Botoom 5 rows \n",df.tail())

	if(choice==2):
		print(df.describe().T)

	if(choice==3):
		df['zn']=df['zn'].astype('int8')

	if(choice==4):

		new=[ 'rm','age','dis']
		fig,axes=plt.subplots(2,2)
		fig.suptitle("Before outliers remove")

		sns.boxplot(data=df,x='rm',ax=axes[0,0])
		sns.boxplot(data=df,x='age',ax=axes[0,1])
		sns.boxplot(data=df,x='dis',ax=axes[1,0])
		# sns.boxplot(data=df,x='rm',axes=[1,1])
		plt.show()

		for val in new:
			df=removeout(df,val)
		fig,axes=plt.subplots(2,2)
		sns.boxplot(data=df,x='rm',ax=axes[0,0])
		sns.boxplot(data=df,x='age',ax=axes[0,1])
		sns.boxplot(data=df,x='dis',ax=axes[1,0])
		plt.show()

	if(choice==5):
		sns.heatmap(df.corr(),annot=True)
		plt.show()
		sns.boxplot(df['rm'])
		plt.show()
		print(df.corr())
	if(choice==6):
		from sklearn.model_selection import train_test_split
		X=df[['rm','lstat']]
		y=df['medv']

		xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=11)
		
		print(xtest.shape)
		print(xtrain.shape)
		print(ytrain.shape)
		print(ytest.shape)


		print(xtrain)
		print(xtest)
		print(ytrain)
		print(ytest)

		from sklearn.linear_model import LinearRegression
		model=LinearRegression().fit(xtrain,ytrain)
		output=model.predict(xtest)
		print(output)

		from sklearn.metrics  import mean_absolute_error

		print(mean_absolute_error(ytest,output))

		print('Score >>>>>>',model.score(xtest,ytest))