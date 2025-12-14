import re
import os
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Конфигурация
TOKEN = os.getenv("BOT_TOKEN")
MODEL_DIR = "model"
DB_PATH = "purchases.db"
MAX_LEN = 128

# Логгирование
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Настройка бота
bot = Bot(token=TOKEN)
dp = Dispatcher()
router = Router()

# Загрузка модели NER
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()
label_list = ["O", "B", "I"]


def init_db():
    """Инициализация базы данных для хранения информации о покупках"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS purchases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            product TEXT,
            raw_line TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized.")


init_db()


def clean_text(text, max_digit_len=5):
    """Очистка чеков"""
    stopwords_set = {
        "ст", "см", "гр", "г", "л", "мл", "кг", "шт", "мкм", "м", "lm",
    }
    text = text.lower()
    # Удаляем точки и запятые
    text = re.sub(r'[.,]', ' ', text)
    # Удаляем все числа
    text = re.sub(r'\d+', ' ', text)
    # Удаляем числа с единицами измерения (с пробелом и без)
    units = r"(ст|см|гр|г|л|мл|кг|шт|мкм|м|lm)"
    text = re.sub(rf"\d+\s?{units}", ' ', text)
    # Удаляем скобки и содержимое
    text = re.sub(r'\([^)]*\)|\[[^\]]*\]|\{[^}]*\}', ' ', text)
    # Удаляем процентное содержание
    text = re.sub(r'\d+(\.\d+)?%', ' ', text)
    # Удаляем мусорные символы
    text = re.sub(r'["\'\/°€#$%=]+', ' ', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    # Токенизация
    tokens = text.split()
    # Удаляем одиночные символы и цифры
    tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]
    # Удаляем стоп-слова
    if stopwords_set is not None:
        tokens = [t for t in tokens if t not in stopwords_set]
    return tokens


def ner_predict(text: str):
    """Нахождение товаров в чеке"""
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    )

    offsets = encoded["offset_mapping"][0].tolist()
    outputs = model(**{k: v for k, v in encoded.items() if k != "offset_mapping"})
    preds = outputs.logits.argmax(-1)[0].tolist()

    spans = []
    current = None

    for i, p in enumerate(preds):
        start, end = offsets[i]
        if (start, end) == (0, 0):
            continue
        label = label_list[p]

        if label == "B":
            if current:
                spans.append(current)
            current = {"start": start, "end": end}
        elif label == "I" and current:
            current["end"] = end
        else:
            if current:
                spans.append(current)
                current = None

    if current:
        spans.append(current)

    products = []
    for s in spans:
        raw = text[s["start"]:s["end"]].strip().lower()
        tokens = clean_text(raw)
        if tokens:
            products.append(" ".join(tokens))

    if not products:
        tokens = clean_text(text)
        if tokens:
            products.append(" ".join(tokens))

    return products


def log_purchase(user_id: int, product: str, raw_line: str):
    """Добавление покупок в БД """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO purchases (user_id, product, raw_line) VALUES (?, ?, ?)",
            (str(user_id), product, raw_line)
        )
        conn.commit()
        conn.close()
        logger.info(f"User {user_id} purchased '{product}' from line '{raw_line}'")
    except Exception as e:
        logger.error(f"Failed to log purchase: {e}")


def get_stats(user_id: int, days: int):
    """Получение статистики"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cutoff = datetime.now() - timedelta(days=days)
        cur.execute(
            "SELECT product, COUNT(*) FROM purchases WHERE user_id = ? AND ts > ? GROUP BY product",
            (str(user_id), cutoff)
        )
        rows = cur.fetchall()
        conn.close()
        logger.info(f"Fetched stats for user {user_id} for last {days} days")
        return rows
    except Exception as e:
        logger.error(f"Failed to fetch stats: {e}")
        return []


@router.message(F.text == "/help")
async def cmd_help(msg: Message):
    """/help"""
    logger.info(f"/help requested by user {msg.from_user.id}")
    help_text = (
        "Отправьте строки из чека — я выделю товары и сохраню их.\n"
        "Поддерживаются несколько строк через Enter.\n"
        "Пример чековой строки: «Хлеб Домашний б/др.600г Рижский»\n\n"
        "Команды статистики:\n"
        "/day — покупки за последние сутки\n"
        "/week — покупки за последнюю неделю\n"
        "/month — покупки за последний месяц"
    )
    await msg.answer(help_text)


@router.message(F.text.startswith("/start"))
async def cmd_start(msg: Message):
    logger.info(f"/start requested by user {msg.from_user.id}")
    await msg.answer("Привет! Отправьте строки из чека — я выделю товары и сохраню их.\n"
                     "Поддерживаются несколько строк через Enter.\n"
                     "Пример чековой строки: «Хлеб Домашний б/др.600г Рижский»\n\n"
                     "Команды статистики:\n"
                     "/day — покупки за последние сутки\n"
                     "/week — покупки за последнюю неделю\n"
                     "/month — покупки за последний месяц\n\n"
                     "/help — помощь"
                     )


@router.message(F.text == "/day")
async def cmd_day(msg: Message):
    rows = get_stats(msg.from_user.id, 1)
    if not rows:
        await msg.answer("За последние сутки покупок не найдено.")
        return
    text = "\n".join(f"{p}: {c} раз" for p, c in rows)
    await msg.answer("Покупки за день:\n" + text)


@router.message(F.text == "/week")
async def cmd_week(msg: Message):
    rows = get_stats(msg.from_user.id, 7)
    if not rows:
        await msg.answer("За неделю ничего не найдено.")
        return
    text = "\n".join(f"{p}: {c} раз" for p, c in rows)
    await msg.answer("Покупки за неделю:\n" + text)


@router.message(F.text == "/month")
async def cmd_month(msg: Message):
    rows = get_stats(msg.from_user.id, 30)
    if not rows:
        await msg.answer("За месяц ничего не найдено.")
        return
    text = "\n".join(f"{p}: {c} раз" for p, c in rows)
    await msg.answer("Покупки за месяц:\n" + text)


@router.message(F.text)
async def process_text(msg: Message):
    user_id = msg.from_user.id
    logger.info(f"Received message from user {user_id}: {msg.text}")

    lines = [l.strip() for l in msg.text.split("\n") if l.strip()]
    all_products = []

    for line in lines:
        products = ner_predict(line)
        for p in products:
            log_purchase(user_id, p, line)
            all_products.append(p)

    await msg.answer("Зафиксировано:\n" + "\n".join(all_products))


async def main():
    dp.include_router(router)
    logger.info("Bot started polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
