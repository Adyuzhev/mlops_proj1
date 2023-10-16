import yaml
import sys
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


params = yaml.safe_load(open("params.yaml"))["scaler"]

train_input = sys.argv[1]
test_input = sys.argv[2]

train_output = os.path.join("data", "stage2", "train.csv")
os.makedirs(os.path.join("data", "stage2"), exist_ok=True)
test_output = os.path.join("data", "stage2", "test.csv")
os.makedirs(os.path.join("data", "stage2"), exist_ok=True)

df_train = pd.read_csv(train_input)
X_train = df_train.drop(columns='target')
y_train = df_train['target'].reset_index(drop=True)

df_test = pd.read_csv(test_input)
X_test = df_test.drop(columns='target')
y_test = df_test['target'].reset_index(drop=True)

scaler = params["chosen_scaler"]

if scaler == "MinMaxScaler":
    mm_scaler = MinMaxScaler()
    X_train_scaled = mm_scaler.fit_transform(X_train)
    X_test_scaled = mm_scaler.transform(X_test)

elif scaler == "StandardScaler":
    st_scaler = StandardScaler()
    X_train_scaled = st_scaler.fit_transform(X_train)
    X_test_scaled = st_scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled)
X_test_scaled = pd.DataFrame(X_test_scaled)

pd.concat([X_train_scaled, y_train], axis=1).to_csv(train_output, index=None)
pd.concat([X_test_scaled, y_test], axis=1).to_csv(test_output, index=None)