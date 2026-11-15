import os
import numpy as np, pandas as pd
import locomotif.locomotif as locomotif
import locomotif.visualize as visualize
import matplotlib.pyplot as plt

df = pd.read_csv('./tabular_result.csv', header=None)

single_col = df.stack().reset_index(drop=True).to_frame(name='Values')

ts = single_col.to_numpy()

ts_norm = (ts - np.mean(ts, axis=None)) / np.std(ts, axis=None)

plt.figure(figsize=(8, 3))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Time segments", fontsize=14)
plt.ylabel("Labels (normalized)", fontsize=14)
plt.plot(ts_norm)
plt.savefig("./patterns/Subsequences.png",bbox_inches = "tight")