import pandas as pd
import numpy as np
import csv
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, f1_score, recall_score, average_precision_score, precision_recall_curve

data = pd.read_csv('csv_result-seismic-bumps.csv', index_col='id', parse_dates=True)
data.head()
data.tail()

#
# X = data[['genergy','gpuls','gdenergy','gdpuls','energy','maxenergy']].values
# y = data[['nbumps']].values

cat_cols = ['seismic','seismoacoustic','shift','ghazard']

le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

X = data[['seismic','seismoacoustic','shift','ghazard','genergy','gpuls','gdenergy','gdpuls','nbumps','nbumps2','nbumps3','nbumps4','nbumps5','nbumps6','nbumps7','nbumps89','energy','maxenergy']].values
y = data[['class']].values

X_train, X_test, y_train, y_test = train_test_split(X, y)


scaler = StandardScaler()
scaler.fit(X_train, y_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

env_labels = {
'activation': ['relu'],
'solver': ['lbfgs', 'sgd', 'adam'],
'alpha':(10**(-i) for i in range(8)),
'learning_rate': ['constant', 'invscaling', 'adaptive']
}


csv_file = open('mlp_test_{}.csv'.format(datetime.datetime.now().isoformat()), 'w')
fwriter = csv.writer(csv_file, dialect='unix', delimiter=' ')

csv_labels = list(env_labels.keys()) + ['precision', 'recall', 'f1-score']
fwriter.writerow(csv_labels)

for ac in env_labels['activation']:
    for slv in env_labels['solver']:
        for a in env_labels['alpha']:
            for lr in env_labels['learning_rate']:
                info = [ac, slv, str(a), lr]
                print(','.join(info))
                mlp = MLPClassifier(activation=ac, solver=slv, alpha=a, learning_rate=lr)
                mlp.fit(X_train_scaled, y_train)
                y_pred = mlp.predict(X_test_scaled)
                info += classification_report(y_test, y_pred).split('\n')[-2].split('      ')[1:-1]
                fwriter.writerow(info)

# ct_sc = OneHotEncoder()
# ct_sc.fit(X_train.reshape(-1, 1), y_train)
# y_train_scaled = ct_sc.transform(y_train)
# y_test_scaled = ct_sc.transform(y_test)

# mlp = MLPClassifier(hidden_layer_sizes=(100,15), max_iter=100, warm_start=True)

# mlp.fit(X_train_scaled,y_train)

# y_pred = mlp.predict(X_train_scaled)

# print(classification_report(y_train, y_pred))

# print(average_precision_score(y_train.reshape(1,-1)[0], y_pred.reshape(1,-1)[0], average='weighted'))

pass