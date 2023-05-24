# DSBDA Starter pack

1. Import libraries (pandas, numpy, seaborn, matplotlib.pyplot)
2. Read the dataset as CSV ( df = read_csv("<filename>"))
3. Display information about dataset (info, shape, size, columns, dtypes, head(), tail(), sample(5), use .T to perform transform and also df.describe() for printing statistical info)
4. Count the not available values and make sum df.isna().sum()








<!-- Others -->
1. Accessing one perticular row and selected list of row 
      - df.loc[<row_label>]
            ex. 1.df[1]
            ex. 2.df[[1,2,3]]  (You can pass the array of labels for multiple row)
      - df.iloc[<row_index>]
            ex. 1. df.iloc[1]             (This is index not label)
            ex.2. df.iloc[[1,2,3,4]]      (Array of indexes)
      - You can also use df.loc[:5] or df.iloc[:5] for accessing the rows
2. Accessing the perticular row and selected list of columns
      - df["gender"]                      ( Get single column )
      - df[["gender", "hsc_p"]]           ( Passing array of column label's)