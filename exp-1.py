

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Missing values:")
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)

scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)

print("Normalized data:")
print(df_normalized)