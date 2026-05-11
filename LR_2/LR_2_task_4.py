from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

input_file = 'income_data.txt'

X_raw = []
count_class1 = 0
count_class2 = 0
max_datapoints = 15000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line.strip().split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X_raw.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X_raw.append(data)
            count_class2 += 1


X_raw = np.array(X_raw)

label_encoder = []
X_encoded = np.empty(X_raw.shape)

for i, item in enumerate(X_raw[0]):
    if item.isdigit():
        X_encoded[:, i] = X_raw[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X_raw[:, i])
        label_encoder.append(le)

X = X_encoded[:, :-1].astype(float)
y = X_encoded[:, -1].astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', max_iter=5000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('CART', DecisionTreeClassifier(random_state=42)))
models.append(('NB', GaussianNB()))
models.append(('SVM', LinearSVC(max_iter=5000, dual='auto', random_state=42)))

results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} (±{cv_results.std():.4f})")

pyplot.figure(figsize=(10, 6))
pyplot.boxplot(results, tick_labels=names)
pyplot.title(
    'Порівняння алгоритмів класифікації (дані про доходи)', fontsize=14)
pyplot.ylabel('Точність (Accuracy)', fontsize=12)
pyplot.xlabel('Алгоритм', fontsize=12)
pyplot.xticks(rotation=45)
pyplot.grid(True, alpha=0.3)
pyplot.tight_layout()
pyplot.savefig('income_algorithm_comparison.png', dpi=150)
pyplot.show()

best_model = KNeighborsClassifier(n_neighbors=5)
best_model.fit(X_train, Y_train)
predictions = best_model.predict(X_validation)

print(
    f"\nТочність на тестовому наборі: {accuracy_score(Y_validation, predictions)*100:.2f}%")
print(f"\nМатриця помилок:")
print(confusion_matrix(Y_validation, predictions))
print(f"\nЗвіт про класифікацію:")
print(classification_report(Y_validation, predictions))
