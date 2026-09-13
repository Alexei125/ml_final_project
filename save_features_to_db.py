import pandas as pd
import pickle
import json
from sqlalchemy import create_engine

DB_USER = "robot-startml-ro"
DB_PASSWORD = "pheiph0hahj1Vaif"
DB_HOST = "postgres.lab.karpov.courses"
DB_PORT = 6432
DB_NAME = "startml"
CONN_STR = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print("Загрузка данных из CSV...")
df_users = pd.read_csv("users.csv")
df_posts = pd.read_csv("posts.csv")

print("Загрузка энкодеров и TF-IDF...")
with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

def safe_transform(le, values):
    values_str = values.astype(str)
    known = set(le.classes_)
    unknown_mask = ~values_str.isin(known)
    if unknown_mask.any():
        print(f"  Заменяем {unknown_mask.sum()} неизвестных значений на '{le.classes_[0]}'")
        values_str = values_str.where(~unknown_mask, le.classes_[0])
    return le.transform(values_str)

print("Подготовка признаков пользователей...")
categorical_cols = ['gender', 'country', 'city', 'exp_group', 'os', 'source']
df_user_features = df_users[['user_id'] + categorical_cols + ['age']].copy()
for col in categorical_cols:
    df_user_features[col] = df_user_features[col].fillna('unknown')
for col in categorical_cols:
    le = encoders[col]
    df_user_features[col] = safe_transform(le, df_user_features[col])
    # Не переименовываем – оставляем имена как есть
print(f"Признаки пользователей: {df_user_features.shape}")

print("Подготовка признаков постов...")
df_post_features = df_posts[['post_id', 'text', 'topic']].copy()
df_post_features['topic'] = df_post_features['topic'].fillna('unknown')
topic_encoder = encoders['topic']
df_post_features['topic'] = safe_transform(topic_encoder, df_post_features['topic'])  # оставляем имя 'topic'

print("Применение TF-IDF к текстам постов...")
post_texts = df_post_features['text'].fillna('')
tfidf_matrix = tfidf.transform(post_texts)
tfidf_feature_names = tfidf.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df_post_features.index)
df_post_features = pd.concat([
    df_post_features[['post_id', 'topic']],
    tfidf_df
], axis=1)
print(f"Признаки постов: {df_post_features.shape}")

PREFIX = "aleksej_rozhdestvenskij_rsp8375_"   # замените на свой логин
engine = create_engine(CONN_STR)

table_users = f"{PREFIX}user_features"
table_posts = f"{PREFIX}post_features"
print(f"Сохранение таблицы '{table_users}' в БД...")
df_user_features.to_sql(table_users, engine, if_exists='replace', index=False, method='multi')
print(f"Сохранение таблицы '{table_posts}' в БД...")
df_post_features.to_sql(table_posts, engine, if_exists='replace', index=False, method='multi')

print("Сохранение порядка признаков в БД...")
with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
feature_order_str = json.dumps(feature_names)
df_order = pd.DataFrame({"feature_order": [feature_order_str]})
table_order = f"{PREFIX}feature_order"
df_order.to_sql(table_order, engine, if_exists='replace', index=False)

print("✅ Готово! Все данные сохранены в базу данных.")