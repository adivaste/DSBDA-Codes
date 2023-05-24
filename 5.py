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

	df=df[(df[var]>=low)&(df[var]<=high)]
	return df
def Display(ytest,y_pre):
	from sklearn.metrics import confusion_matrix
	cm=confusion_matrix(ytest,y_pre)
	print(cm)

	sns.heatmap(cm,annot=True)
	plt.show()



df=pd.read_csv('social.csv')
print("Dataset Loaded succesfully....")
print(df['Purchased'])
choice=1
while(choice!=10):
	print("----------------Menu------------------")
	print("1.Information oF Dataset")
	print("2.Statistical Information of data set ")
	print("3.Null values in Dataset")
	print("4.Remove outliers ")
	print("5.Encodeing using label encoding ")
	print("6.Tarin test split and apply logistic regeression ")


	choice=int(input("Enter Your choice : "))

	if(choice==1):
		print("Information of Dataset \n",df.info)
		print("Columns In Dataset \n",df.columns)
		print("Shape of Dataset \n",df.shape)
		print("First 5 rows \n",df.head())

	if(choice==2):
		print("Statistical Information of Dataset \n",df.descride())

	if(choice==3):
		print(df.isna().sum())
		df['EstimatedSalary']=df['EstimatedSalary'].fillna(df['EstimatedSalary'].mean())
		print(df.isna().sum())


	if(choice==4):
		new=['Age','EstimatedSalary']
		print(df['Age'].dtype)
		sns.boxplot(data=df,x='Age')
		plt.title('Before removing outlier of Age ')
		plt.show()
		sns.boxplot(data=df,x='EstimatedSalary')
		plt.title('Before removing outlier of EstimatedSalary')
		plt.show()
		for val in new:
			# sns.boxplot(val)
			removeout(df,val)
			# sns.boxplot(val)

		# sns.boxplot('Age')
		sns.boxplot(data=df,x='EstimatedSalary')
		plt.title('After removing outlier of EstimatedSalary')
		sns.boxplot(data=df,x='Age')
		plt.title('After removing outlier of Age')
		plt.show()

	if(choice==5):
		print(df['Gender'])
		df['Purchased']=df['Purchased'].astype('category')
		df['Purchased']=df['Purchased'].cat.codes
		print(df['Purchased'])
		print()
	if(choice==6):
		X=df[['Age','EstimatedSalary']]
		y=df['Purchased']
		from sklearn.model_selection import train_test_split
		xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20,random_state=0)
		print("xtrain",xtrain)
		print("ytrain",ytrain)

		from sklearn.linear_model import LogisticRegression
		mn=LogisticRegression()
		mn.fit(xtrain,ytrain)

		print("Model Score >>> ",mn.score(xtest,ytest))

		y_pre=mn.predict(xtest)

		print("Y_ test",ytest)
		print("Y_predicted",y_pre)

		print(Display(ytest,y_pre))

	if(choice==7):
		X=df[['Age','EstimatedSalary']]
		y=df['Purchased']

		from sklearn.model_selection import train_test_split

		xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20,random_state=11)

		from sklearn.preprocessing import StandardScaler

		sc_X=StandardScaler()

		xtrain=sc_X.fit_transform(xtrain)
		xtest=sc_X.fit_transform(xtest)

		from sklearn.linear_model import LogisticRegression

		log=LogisticRegression()
		log.fit(xtrain,ytrain)
		pre=log.predict(xtest)
		
		print("Score>>> ",log.score(xtest,ytest))
		Display(ytest,pre)
		new_input =[[1.92295008,0],[26,35000],[38,50000],[36,144000],[40,61000]]
		new_output=log.predict(new_input)
		# print("Score>>>>",log.score())
		print(new_output)
		Display(new_input,new_output)

		# print("Score>>>>",log.score(xtest,ytest))

	# if(choice==8):
	# 	from sklearn.linear_model import LogisticRegression
		
	# 	LogisticRegression()fit()
	# 	new_output=LogisticRegression().predict(new_input)
	# 	print(new_input,new_output)
		# log=LogisticRegression()
		# log.fir
		# log.predict(input)
		# print("Score>>>> " ,log.score(input,ytest))