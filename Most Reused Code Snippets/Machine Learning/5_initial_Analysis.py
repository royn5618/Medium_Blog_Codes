def assess_categories(X):
  for col in df.columns:
    print("-----------------------------")
    print("{} : {} unique values".format(col, len(df[col].unique()))
    if len(df[col].unique()) > X:
          print(df[col].value_counts())
    print("-----------------------------")
