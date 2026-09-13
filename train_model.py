import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. Загрузка данных
# ------------------------------------------------------------
print("Загрузка обучающих и тестовых данных...")
X_train = pd.read_pickle("models/X_train.pkl")
X_test = pd.read_pickle("models/X_test.pkl")
y_train = pd.read_pickle("models/y_train.pkl")
y_test = pd.read_pickle("models/y_test.pkl")

# Загружаем идентификаторы для расчёта hitrate
train_ids = pd.read_pickle("models/train_ids.pkl")
test_ids = pd.read_pickle("models/test_ids.pkl")

print(f"X_train shape: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test: {y_test.shape}")

# ------------------------------------------------------------
# 2. Удаляем дублирующиеся столбцы (оставляем первое вхождение)
# ------------------------------------------------------------
print("Удаление дублирующихся столбцов...")
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]
print(f"После удаления дубликатов: X_train shape: {X_train.shape}")

# ------------------------------------------------------------
# 3. Обучение модели
# ------------------------------------------------------------
model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

print("Обучение модели...")
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 4. Оценка качества
# ------------------------------------------------------------
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n=== Метрики классификации на тесте ===")
print(f"ROC-AUC:  {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision:{precision:.4f}")
print(f"Recall:   {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# ------------------------------------------------------------
# 5. Расчёт hitrate@5
# ------------------------------------------------------------
def compute_hitrate_at_k(user_ids, post_ids, y_true, y_pred_prob, k=5):
    df = pd.DataFrame({
        'user_id': user_ids,
        'post_id': post_ids,
        'y_true': y_true,
        'pred_prob': y_pred_prob
    })
    df_sorted = df.sort_values(['user_id', 'pred_prob'], ascending=[True, False])
    top_k = df_sorted.groupby('user_id').head(k)
    hit = top_k.groupby('user_id')['y_true'].max()
    hitrate = hit.mean()
    return hitrate

hitrate_5 = compute_hitrate_at_k(
    test_ids['user_id'].values,
    test_ids['post_id'].values,
    y_test.values,
    y_pred_prob,
    k=5
)

print(f"\n=== Hitrate@5 на тесте: {hitrate_5:.4f} ===")

# ------------------------------------------------------------
# 6. Сохранение модели и метрик
# ------------------------------------------------------------
os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

metrics = {
    'roc_auc': roc_auc,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'hitrate@5': hitrate_5
}
with open("models/metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("\nМодель и метрики сохранены в папку 'models/'")
print("Готово! Теперь можно переходить к сервису.")