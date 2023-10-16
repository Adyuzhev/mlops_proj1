from sklearn import datasets
import pandas as pd


iris = datasets.load_iris()
x = iris['data']
y = iris['target']

df = pd.DataFrame(x)
df['target'] = y
df.to_csv('../../data/raw/raw_data.csv', index=False)
