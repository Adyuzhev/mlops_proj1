
import os
import sys
import pickle
import json
import pandas as pd


f_input_test = sys.argv[1]
f_input_model = sys.argv[2]

f_output_score = os.path.join("evaluate", "score.json")
os.makedirs(os.path.join("evaluate"), exist_ok=True)

df_test = pd.read_csv(f_input_test)
X_test = df_test.drop(columns='target')
y_test = df_test['target']

model = pickle.load(open(f_input_model, 'rb'))
score = model.score(X_test, y_test)

with open(f_output_score, "w") as fd:
    json.dump({"score": score}, fd)
