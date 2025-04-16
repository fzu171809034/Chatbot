import sqlite3
import datetime
from config import DB_PATH

conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# 初始化表结构
with conn:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT,
            ai_output TEXT,
            timestamp TEXT
        )
    ''')

def save_message(user_input, ai_output):
    with conn:
        conn.execute(
            "INSERT INTO chat (user_input, ai_output, timestamp) VALUES (?, ?, ?)",
            (user_input, ai_output, datetime.datetime.now().isoformat())
        )


def load_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_input, ai_output, timestamp 
        FROM (
        SELECT * FROM chat ORDER BY id DESC LIMIT 10
        )
        ORDER BY id ASC

    """)
    rows = cursor.fetchall()
    conn.close()
    history_text = "\n\n".join(
        [f"[{row[2]}]\n你：{row[0]}\nAI：{row[1]}" for row in rows]
    )
    return history_text or "暂无记录"