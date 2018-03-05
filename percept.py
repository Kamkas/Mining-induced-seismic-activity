import pandas as pd
import numpy as np
import csv
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

data = pd.read_csv('csv_result-seismic-bumps.csv', index_col='id', parse_dates=True)

"""
	seismic,seismoacoustic,shift,ghazard,class
    id,genergy,gpuls,gdenergy,gdpuls,nbumps,nbumps2,nbumps3,nbumps4,nbumps5,nbumps6,nbumps7,nbumps89,energy,maxenergy

    @attribute seismic {a,b,c,d}
	@attribute seismoacoustic {a,b,c,d}
	@attribute shift {W, N}
	@attribute genergy real
	@attribute gpuls real
	@attribute gdenergy real
	@attribute gdpuls real
	@attribute ghazard {a,b,c,d}
	@attribute nbumps real
	@attribute nbumps2 real
	@attribute nbumps3 real
	@attribute nbumps4 real
	@attribute nbumps5 real
	@attribute nbumps6 real
	@attribute nbumps7 real
	@attribute nbumps89 real
	@attribute energy real
	@attribute maxenergy real
	@attribute class {1,0}

"""

data.head()
data.tail()
# X = data['gpuls'].reshape(-1, 1)
# y = data['seismic']

# X = data[['genergy','gpuls','gdenergy','gdpuls','energy','maxenergy']].values
cat_cols = ['seismic','seismoacoustic','shift','ghazard']

le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

X = data[['seismic','seismoacoustic','shift','ghazard','genergy','gpuls','gdenergy','gdpuls','nbumps','nbumps2','nbumps3','nbumps4','nbumps5','energy','maxenergy']].values
y = data[['class']].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
scaler.fit(X_train, y_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

env_labels = {'penalty': [None,'l1','l2','elasticnet'], 'alpha':(10**(-i) for i in range(8)), 'class_weight':[None, 'balanced']}

csv_file = open('ppn_test_{}.csv'.format(datetime.datetime.now().isoformat()), 'w')
fwriter = csv.writer(csv_file, dialect='unix', delimiter=' ')

csv_labels = list(env_labels.keys()) + ['precision', 'recall', 'f1-score']
fwriter.writerow(csv_labels)

for pn in [None,'l1','l2','elasticnet']:
    for a in env_labels['alpha']:
        for clsw in env_labels['class_weight']:
            info = [str(pn), str(a), str(clsw)]
            print(''.join(info))
            clf = Perceptron(penalty=pn, alpha=a, class_weight=clsw)
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            info += classification_report(y_test, y_pred).split('\n')[-2].split('      ')[1:-1]
            fwriter.writerow(info)

# csv_file.close()

# clf = Perceptron(alpha=.0001, n_iter=100)

    

# clf.fit(X_train_scaled, y_train)

# y_pred = clf.predict(X_test_scaled)

# print('Perceptron properties:\n')
# for k,v in clf.__dict__:
#     print("{0}: {1}".format(k, v))
# print(classification_report(y_test, y_pred))

pass