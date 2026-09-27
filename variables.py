import pandas as pd

df = pd.read_csv('Datos.csv') 

corr = df.corr()
mask = (corr.abs() > 0.5) & (corr != 1.0)
filtered_corr = corr.where(mask).dropna(how='all').dropna(axis=1, how='all')

print(filtered_corr)
