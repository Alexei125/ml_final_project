import os
import pickle
import json
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from database import postgres_connection
from schema import PostGet

# === Вспомогательные функции ===
def load_sql(query: str, dtypes: Dict[str, Any] = None) -> pd.DataFrame:
    conn = postgres_connection()
    try:
        df = pd.read_sql(query, conn, dtype=dtypes)
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка при выполнении SQL-запроса: {e}\nЗапрос: {query}") from e
    finally:
        conn.close()
    return df

def load_model(model_path: str = "model.pkl"):
    if os.environ.get("IS_LMS", "0") == "1":
        model_path = os.environ["MODEL_PATH"]
    logger.info(f"Загрузка модели из файла {model_path}...")
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Файл модели не найден: {model_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка при загрузке модели: {e}") from e
    logger.success("Модель успешно загружена")
    return model

# === Конфигурация ===
PREFIX = "aleksej_rozhdestvenskij_rsp8375_"   # ЗАМЕНИТЕ НА СВОЙ ЛОГИН

# === Загрузка ресурсов при старте ===
logger.info("Инициализация сервиса...")
app = FastAPI()

# 1. Модель
model = load_model("models/model.pkl")

# 2. Признаки пользователей
logger.info("Загрузка признаков пользователей...")
user_features = load_sql(f"SELECT * FROM {PREFIX}user_features")
user_features.set_index("user_id", inplace=True)
user_features.rename(columns=lambda x: x.replace('_enc', ''), inplace=True)

# 3. Признаки постов
logger.info("Загрузка признаков постов...")
post_features = load_sql(f"SELECT * FROM {PREFIX}post_features")
post_features.set_index("post_id", inplace=True)
post_features.rename(columns=lambda x: x.replace('_enc', ''), inplace=True)

# 4. Тексты и темы постов
posts_text = load_sql("SELECT post_id AS id, text, topic FROM public.post_text_df")
posts_text.set_index("id", inplace=True)

# 5. Порядок признаков (из БД)
logger.info("Загрузка порядка признаков из БД...")
order_df = load_sql(f"SELECT feature_order FROM {PREFIX}feature_order")
feature_order_str = order_df.iloc[0]['feature_order']
feature_names = json.loads(feature_order_str)

# Удаляем возможные дубликаты, сохраняя порядок
unique_feature_names = list(dict.fromkeys(feature_names))
if len(unique_feature_names) != len(feature_names):
    logger.warning(f"Были дубликаты в feature_names, теперь {len(unique_feature_names)} уникальных признаков")
    feature_names = unique_feature_names

logger.info(f"Загружено {len(feature_names)} признаков")

logger.success("Сервис успешно инициализирован")

# === Эндпоинт ===
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(user_id: int, dt: datetime, limit: int = 10) -> List[PostGet]:
    if user_id not in user_features.index:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    user_row = user_features.loc[user_id]

    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month
    weekend = 1 if day_of_week >= 5 else 0

    # Копируем признаки постов
    X_pred = post_features.copy(deep=True)

    # Добавляем признаки пользователя
    for col in user_row.index:
        X_pred[col] = user_row[col]

    # Временные признаки
    X_pred["hour"] = hour
    X_pred["day_of_week"] = day_of_week
    X_pred["month"] = month
    X_pred["weekend"] = weekend

    # Длина текста и количество слов
    texts = posts_text.loc[X_pred.index]["text"].fillna("")
    X_pred["text_len"] = texts.str.len()
    X_pred["word_count"] = texts.str.split().str.len()

    # Принудительно приводим к набору колонок из feature_names
    # Удаляем лишние
    X_pred = X_pred[feature_names]

    # Добавляем недостающие колонки с нулями (если вдруг)
    for col in feature_names:
        if col not in X_pred.columns:
            X_pred[col] = 0
            logger.warning(f"Добавлена недостающая колонка {col} с нулями")

    # Переупорядочиваем
    X_pred = X_pred[feature_names]

    # Логируем размер для отладки
    logger.info(f"Размер X_pred: {X_pred.shape}, ожидается {len(feature_names)}")

    # Предсказания (с отключением проверки формы для надёжности)
    probs = model.predict_proba(X_pred, predict_disable_shape_check=True)[:, 1]

    # Топ-N
    top_indices = np.argsort(probs)[-limit:][::-1]

    recs = []
    for idx in top_indices:
        post_id = X_pred.index[idx]
        row = posts_text.loc[post_id]
        recs.append(PostGet(id=post_id, text=row["text"], topic=row["topic"]))

    return recs