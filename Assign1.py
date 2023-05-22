# Importing all libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Read dataset : pd.read.json() || pd.read_excel() || pd.read_sql()  || pd.read_csv()
# DF : DataFrame is the type of object inside the pandas for storing the data
df = pd.read_csv("./sime/placement_data.csv")
print(" :: Placement data loaded into the dataframe")


# Display basic information of the dataset
print(" :: Information about dataframe : ", df.info)              # Consist of : All columns, total rows, data type, non-null count
print(" :: Total data entries : " ,  df.size)                     # Return the total number of value or entries
print(" :: Column Information " ,  df.columns)                    # Return the array of labels || len(df.columns) for size of column size
print(" :: Shape " ,  df.shape);                                  # Return the tuple of Rows and Columns count
print(" :: Show datatypes \n", df.dtypes )


# Display statistical information of dataset
print(" :: Statistical information : " , df.describe())

# Display count of na values
print(" :: Count of NA values : " , df.isna().sum())


# Data type conversion
df['sl_no'] = df['sl_no'].astype('int8')
print(" :: Data type : " , df['sl_no'].dtype)