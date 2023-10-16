
import yaml
import sys
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle


params = yaml.safe_load(open("params.yaml"))["train"]

f_input_train = sys.argv[1]

f_output_model = os.path.join("models", sys.argv[2])
os.makedirs(os.path.join("models"), exist_ok=True)

p_random_state = params["random_state"]
p_penalty = params["penalty"]
p_C = params["C"]
p_fit_intercept = params["fit_intercept"]

df_train = pd.read_csv(f_input_train)
X_train = df_train.drop(columns='target')
y_train = df_train['target']

logreg = LogisticRegression(penalty=p_penalty, C=p_C, fit_intercept=p_fit_intercept, random_state=p_random_state)
logreg.fit(X_train, y_train)

pickle.dump(logreg, open(f_output_model, 'wb'))

