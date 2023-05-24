# ------------------- Assingment No. 1 : Data Wrangling I -------------------

# Importing all libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Read dataset : pd.read.json() || pd.read_excel() || pd.read_sql()  || pd.read_csv()
# DF : DataFrame is the type of object inside the pandas for storing the data
df = pd.read_csv("./placement_data.csv")
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
df['sl_no'] = df['sl_no'].astype('int32')
print(" :: Data type : " , df['sl_no'].dtype)


# Label encoding conversion of categorical data to numerical data : We have the convert the string or object data into numericals only
#  It is used because many ML algo are better at performing numerical calculations, and it also saves the memory and reduce redundancy
print(" :: Encoding using Label Encoding (Cat codes)")
df['gender'] = df['gender'].astype("category")                                  # This will convert the all the values of gender as a category internally and return it
print(":: Data type of gender Changed (to category) ", df.dtypes['gender'])          
                                                                                # Column gender is of a 'category' type now, but we have changed the data yet.
df['gender'] = df['gender'].cat.codes                                           # 'cat.codes' stores the numeric codes for each category derived after converting data row 
                                                                                # into category, and then assigns it, so now our data is in numrical format
print(df['gender'].unique())                                                    # To see the relationship between that categorical codes and values, we can use unique()
print(":: Data types after encoding ", df.dtypes['gender'])


# ::: Extra - You can use pd.factorize() to convert the categorical data to numerical
df['gender'] = pd.factorize(df['gender'])[0]                                    # This will replace the gender column with numerical values. factorize() returns a tuple of
                                                                                # values and unique labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])                                   # This also return the code values as assign it to gender


# Normalization using min max scaling : X_normalized = (X - X.min())/(X.max() - X.min())
# Normalization is the proccess of scaling up and scaling down the values into perticular common range
# for min-max scaling normalization we will scale value in range of 0 and 1 . That will be easier to compare the values
# for the model to compare the values and ML models will do effient way as we have small value or perticular range of values
# Consider the example of height of student in various format, in different dataset
print("Normalization using a Min-Max method :")
df['salary'] = (df['salary'] - df['salary'].min())/(df['salary'].max() - df['salary'].min())
print(df.head().T)                                                              # To transform


# ---------------------------  END  ---------------------------



# Extras : 
'''
============= HELP =============
help(df) and dir(df) to get information about all function and usage

============= DATA TYPES =============

1. Numeric Data types:
    - Non-nullable numeric data types : Which can't handle values like NaN
        :: int64, int32, int16, int8 only
        :: float64, float32 only
    - Nullable 'integer' types : Which can handle values like NaN
        :: Int64, Int32, Int16, Int8 only
2. Boolean data types:
    - Boolean values : Handles True and False values
    - Nullable boolean values
        :: Can Accept the NaN values (We can be replaced further with T&F)
3. String data types:
    :: 'object' data type handles a strings
4. Datetime data types:
    :: 'datetime64' stores the date and time in nanoseconds precision
5. Categorical data :
    :: 'category'  stores the data in category format

- By default for number's data type are int64 and float64
- You can explicitly set the data type at the time of defining 'df'
    df = pd.DataFrame(data, dtype="int32")
- You can select all the columns with specified data types
    df_cat = df.select_dtypes(np.object)       :: These are categorical/object type/string type columns
    df_num = df.select_dtypes(np.number)       :: These are the numeric type columns 

    (For data cleaning we should use numeric mostly data type, so we convert categorical data to numeric )
'''
