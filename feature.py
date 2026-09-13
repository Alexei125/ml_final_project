import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os
import gc

# ------------------------------------------------------------
# 0. Создаём папку для сохранения артефактов (если её нет)
# ------------------------------------------------------------
os.makedirs("models", exist_ok=True)

# ------------------------------------------------------------
# 1. Загрузка данных из CSV
# ------------------------------------------------------------
print("Загрузка данных...")
df_users = pd.read_csv("users.csv")
df_posts = pd.read_csv("posts.csv")
df_feed = pd.read_csv("feed.csv")

print(f"Пользователей: {len(df_users)}")
print(f"Постов: {len(df_posts)}")
print(f"Событий в ленте: {len(df_feed)}")

# ------------------------------------------------------------
# 2. Оптимизация памяти
# ------------------------------------------------------------
print("Оптимизация памяти...")
# Оставляем только нужные колонки в feed (если есть лишние)
# Предполагаем, что в feed есть user_id, post_id, action, timestamp
required_feed_cols = ['user_id', 'post_id', 'action', 'timestamp']
df_feed = df_feed[required_feed_cols].copy()
df_feed['user_id'] = df_feed['user_id'].astype('int32')
df_feed['post_id'] = df_feed['post_id'].astype('int32')
df_users['user_id'] = df_users['user_id'].astype('int32')

# В df_posts переименуем столбец с идентификатором в 'post_id', если он называется иначе
if 'id' in df_posts.columns:
    df_posts.rename(columns={'id': 'post_id'}, inplace=True)
df_posts['post_id'] = df_posts['post_id'].astype('int32')

# ------------------------------------------------------------
# 3. Определение столбца даты (ищем timestamp)
# ------------------------------------------------------------
date_col = None
for col in df_feed.columns:
    if col.lower() in ['timestamp', 'event_date', 'datetime', 'dt', 'created_at']:
        date_col = col
        break

if date_col:
    print(f"Найден столбец даты: {date_col}")
    df_feed['event_date'] = pd.to_datetime(df_feed[date_col], errors='coerce')
    # Удаляем строки с некорректной датой
    df_feed = df_feed.dropna(subset=['event_date'])
else:
    print("Столбец даты не найден. Будет использовано случайное разделение.")
    df_feed['event_date'] = pd.NaT

# ------------------------------------------------------------
# 4. Создание целевой переменной (target)
# ------------------------------------------------------------
df_feed['is_like'] = (df_feed['action'] == 'like').astype(int)

# Для каждой пары (user_id, post_id) определяем, был ли хотя бы один лайк
df_target = df_feed.groupby(['user_id', 'post_id'], as_index=False)['is_like'].max()
df_target.rename(columns={'is_like': 'target'}, inplace=True)

# Определяем дату первого взаимодействия для каждой пары
if date_col:
    df_first_date = df_feed.groupby(['user_id', 'post_id'])['event_date'].min().reset_index(name='first_date')
else:
    # Если даты нет, создаём фиктивную (она не будет использоваться для разделения)
    df_first_date = df_target[['user_id', 'post_id']].copy()
    df_first_date['first_date'] = pd.Timestamp('2023-01-01')

# Объединяем target и first_date
df_pairs = df_target.merge(df_first_date, on=['user_id', 'post_id'], how='inner')
print(f"Получено {len(df_pairs)} пар (user, post)")

# Освобождаем память от больших таблиц
del df_feed, df_target, df_first_date
gc.collect()

# ------------------------------------------------------------
# 5. Присоединение признаков пользователей и постов
# ------------------------------------------------------------
df_pairs = df_pairs.merge(df_users, on='user_id', how='left')
df_pairs = df_pairs.merge(df_posts, left_on='post_id', right_on='post_id', how='left')

# Освобождаем память
del df_users, df_posts
gc.collect()

# ------------------------------------------------------------
# 6. Временные признаки (если есть дата)
# ------------------------------------------------------------
if date_col:
    df_pairs['hour'] = df_pairs['first_date'].dt.hour
    df_pairs['day_of_week'] = df_pairs['first_date'].dt.dayofweek
    df_pairs['month'] = df_pairs['first_date'].dt.month
    df_pairs['weekend'] = df_pairs['day_of_week'].isin([5, 6]).astype(int)
else:
    df_pairs['hour'] = 0
    df_pairs['day_of_week'] = 0
    df_pairs['month'] = 0
    df_pairs['weekend'] = 0

# ------------------------------------------------------------
# 7. Простые признаки текста
# ------------------------------------------------------------
df_pairs['text_len'] = df_pairs['text'].astype(str).str.len()
df_pairs['word_count'] = df_pairs['text'].astype(str).str.split().str.len()

# ------------------------------------------------------------
# 8. Обработка пропусков
# ------------------------------------------------------------
categorical_cols = ['gender', 'country', 'city', 'exp_group', 'os', 'source', 'topic']
for col in categorical_cols:
    df_pairs[col] = df_pairs[col].fillna('unknown')
df_pairs['text'] = df_pairs['text'].fillna('')

# ------------------------------------------------------------
# 9. Разделение на train / test (по времени, если возможно)
# ------------------------------------------------------------
if date_col:
    df_pairs_sorted = df_pairs.sort_values('first_date')
    split_idx = int(0.8 * len(df_pairs_sorted))
    train = df_pairs_sorted.iloc[:split_idx].copy()
    test = df_pairs_sorted.iloc[split_idx:].copy()
    print(f"Разделение по времени: Train {len(train)}, Test {len(test)}")
else:
    train, test = train_test_split(df_pairs, test_size=0.2, random_state=42)
    print(f"Случайное разделение: Train {len(train)}, Test {len(test)}")

# Освобождаем память от df_pairs
del df_pairs, df_pairs_sorted
gc.collect()

# ------------------------------------------------------------
# 10. Кодирование категориальных признаков
# ------------------------------------------------------------
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Обучаем на объединённых данных, чтобы классы совпадали
    combined = pd.concat([train[col], test[col]]).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    encoders[col] = le

# ------------------------------------------------------------
# 11. TF-IDF для текстов (уменьшенный max_features)
# ------------------------------------------------------------
tfidf = TfidfVectorizer(max_features=150, stop_words='english', min_df=5)
train_text = train['text'].fillna('')
test_text = test['text'].fillna('')
tfidf.fit(train_text)   # обучаем только на train

train_tfidf = tfidf.transform(train_text)
test_tfidf = tfidf.transform(test_text)

# Преобразуем в DataFrame
tfidf_feature_names = tfidf.get_feature_names_out()
train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=tfidf_feature_names, index=train.index)
test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=tfidf_feature_names, index=test.index)

# ------------------------------------------------------------
# 12. Формирование X и y (матрицы признаков и целевые переменные)
# ------------------------------------------------------------
drop_cols = ['user_id', 'post_id', 'text', 'first_date']
y_train = train['target']
y_test = test['target']

X_train = train.drop(columns=drop_cols + ['target'], errors='ignore')
X_test = test.drop(columns=drop_cols + ['target'], errors='ignore')

# Добавляем TF-IDF признаки
X_train = pd.concat([X_train, train_tfidf_df], axis=1)
X_test = pd.concat([X_test, test_tfidf_df], axis=1)

# Преобразуем в float32 для экономии памяти
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Сохраняем идентификаторы (user_id, post_id) для расчёта hitrate
train[['user_id', 'post_id']].to_pickle("models/train_ids.pkl")
test[['user_id', 'post_id']].to_pickle("models/test_ids.pkl")

# ------------------------------------------------------------
# 13. Сохранение всех артефактов в папку models/
# ------------------------------------------------------------
print("Сохранение артефактов...")
X_train.to_pickle("models/X_train.pkl")
X_test.to_pickle("models/X_test.pkl")
y_train.to_pickle("models/y_train.pkl")
y_test.to_pickle("models/y_test.pkl")

with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

feature_names = X_train.columns.tolist()
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("Все артефакты сохранены в папку 'models/'")
print("Готово!")