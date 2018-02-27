import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

data = pd.read_csv('csv_result-seismic-bumps.csv', index_col='id', parse_dates=True)
data.head()
data.tail()


X = data[['genergy','gpuls','gdenergy','gdpuls','energy','maxenergy']].values
y = data[['seismic']].values

X_train, X_test, y_train, y_test = train_test_split(X, y)


scaler = StandardScaler()
scaler.fit(X_train, y_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ct_sc = OneHotEncoder()
# ct_sc.fit(X_train.reshape(-1, 1), y_train)
# y_train_scaled = ct_sc.transform(y_train)
# y_test_scaled = ct_sc.transform(y_test)

mlp = MLPClassifier(hidden_layer_sizes=(200,), max_iter=1000, random_state=1)

mlp.fit(X_train_scaled,y_train)

y_pred = mlp.predict(X_train_scaled)

print(classification_report(y_train, y_pred))

pass