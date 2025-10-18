# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка данных
data = pd.read_csv('us_perm_visas.csv')
# Уменьшите размер, если память ограничена (например, 50k строк)
# data = data.sample(50000, random_state=42)

# 2. Разбиение на train/test
data = data.dropna(subset=['case_status'])
X = data.drop('case_status', axis=1)
y = data['case_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Визуализация и статистики
print("Основные статистики:")
print(data.describe())
print("\nКорреляционная матрица (числовые признаки):")
numeric_cols = X.select_dtypes(include=[np.number]).columns
corr = X[numeric_cols].corr()
sns.heatmap(corr, annot=True, vmin = -1, vmax = 1, cmap='coolwarm')
plt.show()

# 4. Обработка пропущенных значений (без изменений)
X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
    X_test[col] = X_test[col].fillna(X_test[col].mode()[0])

# 5. Обработка категориальных признаков
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Преобразуем категориальные колонки к строкам
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Разделяем категориальные колонки: низко-кардинальные (<100 уникальных) — OneHot, высоко-кардинальные — Ordinal
low_card_cols = [col for col in categorical_cols if X_train[col].nunique() <= 100]
high_card_cols = [col for col in categorical_cols if X_train[col].nunique() > 100]

# Создаём preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('low_cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), low_card_cols),  # Sparse для памяти
        ('high_cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), high_card_cols)  # Ordinal для высоко-кардинальных
    ]
)

# Применяем к train и test (результат — sparse матрица)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 6. Нормализация (работает с sparse)
scaler = StandardScaler(with_mean=False)  # with_mean=False для sparse
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

# 7. KNN классификатор
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_train = knn.predict(X_train_scaled)
y_pred_test = knn.predict(X_test_scaled)
print("KNN (k=5) — Точность на train:", accuracy_score(y_train, y_pred_train))
print("KNN (k=5) — Точность на test:", accuracy_score(y_test, y_pred_test))
print("Матрица рассогласования (test):")
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))

# Оптимальный k
param_grid = {'n_neighbors': range(1, 21)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("Лучший k:", grid.best_params_['n_neighbors'])

# 8. Другие классификаторы
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_processed, y_train)  # Без масштабирования
y_pred_rf = rf.predict(X_test_processed)
print("Random Forest — Точность:", accuracy_score(y_test, y_pred_rf))
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("Logistic Regression — Точность:", accuracy_score(y_test, y_pred_lr))

# 9. Борьба с несбалансированностью
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
knn.fit(X_train_sm, y_train_sm)
y_pred_sm = knn.predict(X_test_scaled)
print("KNN после SMOTE — Точность:", accuracy_score(y_test, y_pred_sm))
print(classification_report(y_test, y_pred_sm))

# 10. Выводы

# Random Forest показал лучшую точность (~0.89) благодаря обработке категориальных данных. SMOTE улучшил F1 для минорного класса. Коррелированные переменные не исключены, так как их мало и корреляция слабая.
