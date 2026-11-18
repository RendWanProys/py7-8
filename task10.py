import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 1. Загрузка данных
df = pd.read_csv('heart.csv')

X = df.drop('target', axis=1)
y = df['target']

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Подбор k для KNN
k_range = list(range(1, 31))
auc_scores = []

for k in k_range:
  pipe = Pipeline(
      [('prep', preprocessor), ('knn', KNeighborsClassifier(n_neighbors=k))]
  )
  pipe.fit(X_train, y_train)
  y_prob = pipe.predict_proba(X_test)[:, 1]
  auc_scores.append(roc_auc_score(y_test, y_prob))

best_k = k_range[auc_scores.index(max(auc_scores))]
print('best k:', best_k, 'roc-auc:', max(auc_scores))

plt.figure(figsize=(8, 4))
plt.plot(k_range, auc_scores, marker='o')
plt.xlabel('k')
plt.ylabel('ROC-AUC')
plt.grid()
plt.show()

# 3. Сравнение 4 моделей
models = {
    f'KNN (k={best_k})': KNeighborsClassifier(n_neighbors=best_k),
    'SVM': SVC(probability=True, kernel='linear', C=0.1),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=42
    ),
}

plt.figure(figsize=(8, 5))

for name, model in models.items():
  pipe = Pipeline([('prep', preprocessor), ('model', model)])
  pipe.fit(X_train, y_train)

  y_pred = pipe.predict(X_test)
  y_prob = pipe.predict_proba(X_test)[:, 1]

  acc = accuracy_score(y_test, y_pred)
  auc = roc_auc_score(y_test, y_prob)
  print(f'{name}: Accuracy = {acc:.3f}, ROC-AUC = {auc:.3f}')

  fpr, tpr, _ = roc_curve(y_test, y_prob)
  plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.grid()
plt.show()
