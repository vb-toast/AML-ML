import os
import pandas as pd

df = pd.read_csv(os.path.join('notebook', 'data', 'HI-Small_Trans.csv'))
print(df.shape)
print(df["Is_Laundering"].value_counts())