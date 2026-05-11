import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import sys
sys.stdout.reconfigure(encoding='utf-8')

input_file = 'income_data.txt'

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f:
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line.strip().split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Гаусове ядро (завдання 2)
classifier = SVC(kernel='rbf', max_iter=25000)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("\nГАУСОВЕ ЯДРО:")
print("Accuracy:", round(
    accuracy_score(y_test, y_pred)*100, 2), "%")
print("Precision:", round(precision_score(
    y_test, y_pred, average='weighted')*100, 2), "%")
print("Recall:", round(recall_score(
    y_test, y_pred, average='weighted')*100, 2), "%")
print("F1-score:",
      round(f1_score(y_test, y_pred, average='weighted')*100, 2), "%")
