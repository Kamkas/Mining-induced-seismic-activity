import pandas as pd
import numpy as np
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

data = pd.read_csv('csv_result-seismic-bumps.csv', index_col='id', parse_dates=True)


data.head()
data.tail()
# X = data['gpuls'].reshape(-1, 1)
# y = data['seismic']

X = data[['genergy','gpuls','gdenergy','gdpuls','energy','maxenergy']].values
y = data[['nbumps']].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
scaler.fit(X_train, y_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = Perceptron(alpha=.0001, n_iter=100)



clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

# print('Perceptron properties:\n')
# for k,v in clf.__dict__:
#     print("{0}: {1}".format(k, v))
print(classification_report(y_test, y_pred))

pass