

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

ms = MeanShift()
ms.fit(df_scaled)

print("Cluster Labels:", ms.labels_)

plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=ms.labels_)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Mean Shift Clusters")
plt.show()