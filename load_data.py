import pandas as pd
from database import postgres_connection

def load_data(limit_feed=1_000_000):
    conn = postgres_connection()

    print("Загрузка пользователей...")
    df_users = pd.read_sql("SELECT * FROM public.user_data", con=conn)
    print(f"{len(df_users)} записей")

    print("Загрузка постов...")
    df_posts = pd.read_sql("SELECT * FROM public.post_text_df", con=conn)
    print(f"{len(df_posts)} записей")

    print(f"Загрузка ленты (LIMIT {limit_feed})...")
    query_feed = f"SELECT * FROM public.feed_data LIMIT {limit_feed}"
    df_feed = pd.read_sql(query_feed, con=conn)
    print(f"{len(df_feed)} записей")

    conn.close()
    return df_users, df_posts, df_feed

if __name__ == "__main__":
    users, posts, feed = load_data()
    users.to_csv("users.csv", index=False)
    posts.to_csv("posts.csv", index=False)
    feed.to_csv("feed.csv", index=False)
    print("Данные сохранены в CSV.")
