import numpy as np
import pandas as pd
from parse import parse_text
from matplotlib import pyplot as plt

df1 = pd.read_csv('wine_data_190.csv')
df2 = pd.read_csv('winemag-data-2017-2020.csv')
df2.drop(['taster_photo'], axis=1, inplace=True)

df = df1.append(df2, ignore_index=True)
df.drop(['Unnamed: 0'], axis=1, inplace=True)

df.to_csv('wine_data.csv')
