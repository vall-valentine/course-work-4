import re
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta
import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import requests
import vk_api
from dotenv import load_dotenv
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.utils import get_random_id
from concurrent.futures import ThreadPoolExecutor
import functools
import io
import json

load_dotenv()

# Настройки VK
VK_TOKEN = os.environ.get("VK_TOKEN")
VK_GROUP_ID = int(os.environ.get("VK_GROUP_ID", "0"))

DB_PATH = os.environ.get("DB_PATH", "data/purchases.db")
MODEL_HOST = os.environ.get("MODEL_HOST", "127.0.0.1")
MODEL_PORT = int(os.environ.get("MODEL_PORT", "8000"))

# Настройка Tesseract
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# В Linux/Docker Tesseract устанавливается через apt и доступен по умолчанию

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=2)

# Состояния пользователей
user_states = {}

MODEL_SERVICE_URL = f"http://{MODEL_HOST}:{MODEL_PORT}"
logger.info(f"Модель: {MODEL_SERVICE_URL}")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Таблица покупок
    cur.execute("""
        CREATE TABLE IF NOT EXISTS purchases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            product TEXT,
            raw_line TEXT,
            receipt_id TEXT,
            amount REAL,
            receipt_date TEXT,
            category TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Временные чеки
    cur.execute("""
        CREATE TABLE IF NOT EXISTS temp_receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            receipt_id TEXT,
            products TEXT,
            amount REAL,
            receipt_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Таблица для автоматической разметки
    cur.execute("""
        CREATE TABLE IF NOT EXISTS auto_labeled_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product TEXT,
            category TEXT,
            confidence REAL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


init_db()


def normalize_product_entries(products):
    entries = []
    for product in products:
        if isinstance(product, dict):
            name = str(product.get("product", "")).strip().lower()
            raw_line = str(product.get("raw_line", "")).strip()
        else:
            name = str(product).strip().lower()
            raw_line = ""

        if not name:
            continue

        entries.append({
            "product": name,
            "raw_line": raw_line
        })

    return entries


def product_names(product_entries):
    return [entry["product"] for entry in product_entries]


def parse_stored_products(raw_value: str):
    if not raw_value:
        return []

    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return normalize_product_entries(parsed)
    except (json.JSONDecodeError, TypeError):
        pass

    return normalize_product_entries([item for item in raw_value.split(',') if item])


def save_temp_receipt(user_id: int, receipt_id: str, products: list, amount: float = None, receipt_date: str = None):
    try:
        product_entries = normalize_product_entries(products)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM temp_receipts WHERE user_id = ?", (str(user_id),))
        cur.execute(
            "INSERT INTO temp_receipts (user_id, receipt_id, products, amount, receipt_date) VALUES (?, ?, ?, ?, ?)",
            (
                str(user_id),
                receipt_id,
                json.dumps(product_entries, ensure_ascii=False),
                amount,
                receipt_date,
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка сохранения временного чека: {e}")


def get_temp_receipt(user_id: int) -> tuple:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT receipt_id, products, amount, receipt_date FROM temp_receipts WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (str(user_id),)
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0], parse_stored_products(row[1]), row[2], row[3]
        return None, [], None, None
    except Exception as e:
        logger.error(f"Ошибка получения временного чека: {e}")
        return None, [], None, None


def delete_temp_receipt(user_id: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM temp_receipts WHERE user_id = ?", (str(user_id),))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка удаления временного чека: {e}")


def save_final_receipt(user_id: int, receipt_id: str, products: list, amount: float = None, receipt_date: str = None,
                       category: str = None):
    try:
        product_entries = normalize_product_entries(products)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        for entry in product_entries:
            cur.execute(
                """
                INSERT INTO purchases
                (user_id, product, raw_line, receipt_id, amount, receipt_date, category)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(user_id),
                    entry["product"],
                    entry["raw_line"] or None,
                    receipt_id,
                    amount,
                    receipt_date,
                    category,
                )
            )
        conn.commit()
        conn.close()
        logger.info(f"Сохранен чек {receipt_id} с {len(product_entries)} товарами")
    except Exception as e:
        logger.error(f"Ошибка сохранения чека: {e}")


def delete_last_receipt(user_id: int) -> int:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT receipt_id FROM purchases WHERE user_id = ? ORDER BY ts DESC LIMIT 1",
            (str(user_id),)
        )
        row = cur.fetchone()
        if row:
            receipt_id = row[0]
            cur.execute(
                "DELETE FROM purchases WHERE user_id = ? AND receipt_id = ?",
                (str(user_id), receipt_id)
            )
            deleted = cur.rowcount
            conn.commit()
            conn.close()
            return deleted
        conn.close()
        return 0
    except Exception as e:
        logger.error(f"Ошибка удаления чека: {e}")
        return 0


def get_stats_with_amount(user_id: int, days: int):
    """Получает статистику с суммой трат за период"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cutoff = datetime.now() - timedelta(days=days)

    # Получаем частоту покупок товаров
    cur.execute(
        "SELECT product, COUNT(*) FROM purchases WHERE user_id = ? AND ts > ? GROUP BY product ORDER BY COUNT(*) DESC",
        (str(user_id), cutoff))
    product_stats = cur.fetchall()

    # Получаем общую сумму трат за период
    cur.execute(
        "SELECT SUM(amount) FROM purchases WHERE user_id = ? AND ts > ? AND amount IS NOT NULL",
        (str(user_id), cutoff))
    total_amount = cur.fetchone()[0]

    # Получаем количество чеков за период
    cur.execute(
        "SELECT COUNT(DISTINCT receipt_id) FROM purchases WHERE user_id = ? AND ts > ?",
        (str(user_id), cutoff))
    receipt_count = cur.fetchone()[0]

    conn.close()

    return {
        'products': product_stats,
        'total_amount': total_amount if total_amount else 0,
        'receipt_count': receipt_count if receipt_count else 0
    }


def extract_date_from_ocr(lines: list) -> str:
    """Извлекает дату из OCR текста используя регулярные выражения"""
    date_patterns = [
        r'(\d{2}\.\d{2}\.\d{4})',
        r'(\d{2}\.\d{2}\.\d{2})',
        r'(\d{1,2}\.\d{2}\.\d{4})',
        r'(\d{1,2}\.\d{2}\.\d{2})',
        r'(\d{2}/\d{2}/\d{4})',
        r'(\d{2}/\d{2}/\d{2})',
        r'(\d{4}-\d{2}-\d{2})',
    ]

    for line in lines:
        for pattern in date_patterns:
            match = re.search(pattern, line)
            if match:
                date_str = match.group(1)
                try:
                    if '/' in date_str:
                        parts = date_str.split('/')
                    elif '-' in date_str:
                        parts = date_str.split('-')
                        parts.reverse()
                    else:
                        parts = date_str.split('.')

                    if len(parts) == 3:
                        day = int(parts[0])
                        month = int(parts[1])
                        year = int(parts[2])

                        if year < 100:
                            year += 2000

                        if 1 <= day <= 31 and 1 <= month <= 12 and 2000 <= year <= 2100:
                            return f"{day:02d}.{month:02d}.{year}"
                except:
                    continue

    return None


def extract_amount_from_ocr(lines: list) -> float:
    """Извлекает сумму из OCR текста"""
    amount_patterns = [
        r'[иії][тT][о0][гr]?[о0а]?\s*[:—–-]?\s*([\d\s,]+\.?\d*)',
        r'итог\s*[:—–-]?\s*([\d\s,]+\.?\d*)',
        r'итого\s*[:—–-]?\s*([\d\s,]+\.?\d*)',
        r'всего\s*[:—–-]?\s*([\d\s,]+\.?\d*)',
        r'сумма\s*[:—–-]?\s*([\d\s,]+\.?\d*)',
        r'итог[оа]?\s*[:—–-]?\s*([\d\s,]+\.?\d*)',
        r'([\d\s,]+\.?\d*)\s*[рp]\.',
        r'([\d\s,]+\.?\d*)\s*руб',
        r'([\d\s,]+\.?\d*)\s*рублей',
    ]

    for line in lines:
        line_lower = line.lower().strip()

        for pattern in amount_patterns:
            match = re.search(pattern, line_lower)
            if match:
                amount_str = match.group(1).strip()
                amount_str = amount_str.replace(' ', '').replace(',', '.')
                try:
                    amount = float(amount_str)
                    if amount > 0 and amount < 1000000:
                        logger.info(f"Найдена сумма {amount} в строке: {line}")
                        return amount
                except:
                    continue

    return None


class NERClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def predict(self, lines: list, filter_service: bool = True) -> list:
        try:
            payload = {
                "lines": lines,
                "filter_service": filter_service
            }

            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                predictions = data.get("predictions")
                if isinstance(predictions, list):
                    return normalize_product_entries(predictions)

                return normalize_product_entries(data.get("products", []))
            else:
                logger.error(f"Ошибка модели: {response.status_code}")
                return []

        except requests.exceptions.ConnectionError:
            logger.error("Модель не доступна!")
            return []
        except Exception as e:
            logger.error(f"Ошибка запроса к модели: {e}")
            return []


ner_client = NERClient(MODEL_SERVICE_URL)


def preprocess_receipt_image(image_bytes: bytes):
    """Упрощенная предобработка изображения для OCR - работает стабильно"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    height, width = img.shape[:2]

    # Масштабирование только если изображение слишком маленькое
    if height < 500:
        scale = 1000 / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Конвертация в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Простая бинаризация Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img, binary


def extract_text_lines_tesseract(binary_image) -> list:
    """Простое извлечение текста с базовыми параметрами"""
    pil_img = Image.fromarray(binary_image)

    text = pytesseract.image_to_string(pil_img, lang='rus+eng', config='--oem 3 --psm 6')

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    logger.info(f"OCR распознано {len(lines)} строк")
    if lines:
        logger.debug(f"Первые 3 строки: {lines[:3]}")

    return lines


def ocr_image_sync(image_bytes: bytes, user_id: int) -> tuple:
    """OCR with date and amount extraction."""
    try:
        _, binary = preprocess_receipt_image(image_bytes)
        if binary is None:
            logger.error("Не удалось обработать изображение")
            return [], None, None

        raw_lines = extract_text_lines_tesseract(binary)

        if not raw_lines:
            logger.warning("Текст не распознан")
            return [], None, None

        receipt_date = extract_date_from_ocr(raw_lines)
        amount = extract_amount_from_ocr(raw_lines)

        logger.info(f"Извлечено: дата={receipt_date}, сумма={amount}")

        return raw_lines, receipt_date, amount

    except Exception as e:
        logger.error(f"OCR error: {e}")
        return [], None, None


async def ocr_image(image_bytes: bytes, user_id: int) -> tuple:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        functools.partial(ocr_image_sync, image_bytes, user_id)
    )


def send_message(vk_session, user_id, message, attachment=None):
    """Отправка сообщения пользователю"""
    try:
        vk_session.method('messages.send', {
            'user_id': user_id,
            'message': message,
            'random_id': get_random_id(),
            'attachment': attachment
        })
    except Exception as e:
        logger.error(f"Ошибка отправки сообщения: {e}")


def send_keyboard(vk_session, user_id, message, buttons, one_time=False):
    """Отправка сообщения с клавиатурой"""
    keyboard = {
        "one_time": one_time,
        "buttons": [[{
            "action": {
                "type": "text",
                "label": label
            },
            "color": color
        } for label, color in row] for row in buttons]
    }

    try:
        vk_session.method('messages.send', {
            'user_id': user_id,
            'message': message,
            'random_id': get_random_id(),
            'keyboard': json.dumps(keyboard, ensure_ascii=False)
        })
    except Exception as e:
        logger.error(f"Ошибка отправки клавиатуры: {e}")


def handle_start(vk_session, user_id):
    text = (
        "Привет! Я помогаю сохранять товары из чеков\n\n"
        "Отправь фото чека или вставь текст\n\n"
        "Доступные команды:"
    )
    buttons = [
        [("Статистика за день", "primary")],
        [("Статистика за неделю", "primary")],
        [("Статистика за месяц", "primary")],
        [("Отменить последний чек", "negative")]
    ]
    send_keyboard(vk_session, user_id, text, buttons)


def handle_day(vk_session, user_id):
    stats = get_stats_with_amount(user_id, 1)

    if not stats['products']:
        send_message(vk_session, user_id, "За сегодня покупок нет")
        return

    lines = [
        "Статистика за сегодня",
        f"Всего покупок: {sum(c for _, c in stats['products'])}",
        f"Общая сумма: {stats['total_amount']:.2f} руб.",
        f"Количество чеков: {stats['receipt_count']}",
        "",
        "Топ товаров:"
    ]

    for prod, cnt in stats['products'][:10]:
        lines.append(f"- {prod}: {cnt}")

    send_message(vk_session, user_id, "\n".join(lines))


def handle_week(vk_session, user_id):
    stats = get_stats_with_amount(user_id, 7)

    if not stats['products']:
        send_message(vk_session, user_id, "За неделю покупок нет")
        return

    lines = [
        "Статистика за неделю",
        f"Всего покупок: {sum(c for _, c in stats['products'])}",
        f"Общая сумма: {stats['total_amount']:.2f} руб.",
        f"Количество чеков: {stats['receipt_count']}",
        "",
        "Топ товаров:"
    ]

    for prod, cnt in stats['products'][:10]:
        lines.append(f"- {prod}: {cnt}")

    send_message(vk_session, user_id, "\n".join(lines))


def handle_month(vk_session, user_id):
    stats = get_stats_with_amount(user_id, 30)

    if not stats['products']:
        send_message(vk_session, user_id, "За месяц покупок нет")
        return

    lines = [
        "Статистика за месяц",
        f"Всего покупок: {sum(c for _, c in stats['products'])}",
        f"Общая сумма: {stats['total_amount']:.2f} руб.",
        f"Количество чеков: {stats['receipt_count']}",
        "",
        "Топ товаров:"
    ]

    for prod, cnt in stats['products'][:15]:
        lines.append(f"- {prod}: {cnt}")

    send_message(vk_session, user_id, "\n".join(lines))


def handle_cancel(vk_session, user_id):
    deleted = delete_last_receipt(user_id)
    if deleted > 0:
        send_message(vk_session, user_id, f"Последний чек удален (удалено {deleted} записей)")
    else:
        send_message(vk_session, user_id, "Нет чека для удаления")


def handle_save(vk_session, user_id):
    receipt_id, product_entries, amount, receipt_date = get_temp_receipt(user_id)
    products = product_names(product_entries)

    if not product_entries:
        send_message(vk_session, user_id, "Нет чека для сохранения")
        return

    save_final_receipt(user_id, receipt_id, product_entries, amount, receipt_date)
    delete_temp_receipt(user_id)
    if user_id in user_states:
        del user_states[user_id]

    result = [f"Чек сохранен! ({len(products)} товаров)", ""]
    if amount:
        result.append(f"Сумма: {amount:.2f} руб.")
    if receipt_date:
        result.append(f"Дата: {receipt_date}")
    result.append("")
    for i, p in enumerate(products[:10], 1):
        result.append(f"{i}. {p}")

    send_message(vk_session, user_id, "\n".join(result))

    # Показываем главное меню после сохранения
    handle_start(vk_session, user_id)


def handle_change(vk_session, user_id):
    receipt_id, product_entries, amount, receipt_date = get_temp_receipt(user_id)
    products = product_names(product_entries)

    if not product_entries:
        send_message(vk_session, user_id, "Нет чека для редактирования")
        return

    lines = ["Текущий список товаров:", ""]
    products = product_names(product_entries)
    for i, p in enumerate(products, 1):
        lines.append(f"{i}. {p}")

    if amount:
        lines.append(f"\nСумма: {amount:.2f} руб.")
    if receipt_date:
        lines.append(f"Дата: {receipt_date}")

    lines.append("")
    lines.append("Отправь исправления в формате:")
    lines.append("1 молоко")
    lines.append("3 хлеб")
    lines.append("(или только номер для удаления товара)")
    lines.append("")
    lines.append("После исправлений используй команды Сохранить или Отменить")

    buttons = [
        [("Сохранить", "positive")],
        [("Отменить", "negative")]
    ]
    send_keyboard(vk_session, user_id, "\n".join(lines), buttons, one_time=True)
    user_states[user_id] = 'waiting_for_corrections'


def handle_corrections(vk_session, user_id, text):
    """Обработка исправлений"""
    receipt_id, product_entries, amount, receipt_date = get_temp_receipt(user_id)
    products = product_names(product_entries)

    if not product_entries:
        send_message(vk_session, user_id, "Ошибка: чек не найден")
        if user_id in user_states:
            del user_states[user_id]
        return

    lines = text.strip().split('\n')
    corrections = []
    indices_to_delete = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)(?:\s+(.*))?$', line)
        if match:
            num = int(match.group(1))
            new_product = match.group(2)

            if 1 <= num <= len(products):
                if new_product is None or new_product.strip() == "":
                    indices_to_delete.append(num)
                    logger.info(f"Будет удален товар #{num}: {products[num - 1]}")
                else:
                    corrections.append((num, new_product.strip().lower()))
            else:
                send_message(vk_session, user_id, f"Номер {num} вне диапазона (1-{len(products)})")
                return
        else:
            send_message(vk_session, user_id,
                         f"Неправильный формат: {line}\nНужно: номер товар или просто номер для удаления")
            return

    for num, new_product in corrections:
        product_entries[num - 1]["product"] = new_product
        logger.info(f"Исправлен #{num}: {new_product}")

    for num in sorted(indices_to_delete, reverse=True):
        deleted = product_entries.pop(num - 1)["product"]
        logger.info(f"Удален товар #{num}: {deleted}")

    save_temp_receipt(user_id, receipt_id, product_entries, amount, receipt_date)

    if not product_entries:
        delete_temp_receipt(user_id)
        if user_id in user_states:
            del user_states[user_id]
        send_message(vk_session, user_id, "Все товары удалены. Чек отменен.")
        handle_start(vk_session, user_id)
        return

    result = ["Список обновлен:", ""]
    products = product_names(product_entries)
    for i, p in enumerate(products, 1):
        result.append(f"{i}. {p}")

    if amount:
        result.append(f"\nСумма: {amount:.2f} руб.")
    if receipt_date:
        result.append(f"Дата: {receipt_date}")

    result.append("")
    result.append("Выбери действие:")

    buttons = [
        [("Сохранить", "positive")],
        [("Исправить", "primary")],
        [("Отменить", "negative")]
    ]
    send_keyboard(vk_session, user_id, "\n".join(result), buttons)


def download_vk_photo(vk_session, photo_obj):
    """Скачивание фото из VK"""
    try:
        photo_sizes = photo_obj.get('sizes', [])
        if not photo_sizes:
            return None

        largest_photo = max(photo_sizes, key=lambda x: x.get('width', 0))
        photo_url = largest_photo.get('url')

        if not photo_url:
            return None

        response = requests.get(photo_url, timeout=10)
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"Ошибка скачивания фото: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Ошибка загрузки фото VK: {e}")
        return None


def show_receipt_actions(vk_session, user_id, products, amount, receipt_date):
    result = [f"Найдено {len(products)} товаров:", ""]
    for i, p in enumerate(products, 1):
        result.append(f"{i}. {p}")

    if amount:
        result.append(f"\nСумма: {amount:.2f} руб.")
    if receipt_date:
        result.append(f"Дата: {receipt_date}")

    result.append("")
    result.append("Выбери действие:")

    buttons = [
        [("Сохранить", "positive")],
        [("Исправить", "primary")],
        [("Отменить", "negative")]
    ]
    send_keyboard(vk_session, user_id, "\n".join(result), buttons)


def handle_photo(vk_session, user_id, photo_obj):
    """Обработка фото чека"""
    current_state = user_states.get(user_id)
    if current_state == 'waiting_for_corrections':
        send_message(vk_session, user_id, "Сначала заверши редактирование (Сохранить или Отменить)")
        return

    try:
        send_message(vk_session, user_id, "Распознаю текст...")

        image_bytes = download_vk_photo(vk_session, photo_obj)

        if not image_bytes:
            send_message(vk_session, user_id, "Не удалось скачать фото. Попробуй другое.")
            return

        lines, receipt_date, amount = ocr_image_sync(image_bytes, user_id)

        if not lines:
            send_message(vk_session, user_id, "Не удалось распознать текст. Попробуй другое фото.")
            return

        send_message(vk_session, user_id, f"Распознано {len(lines)} строк, отправляю в нейросеть...")

        product_entries = ner_client.predict(lines, filter_service=True)
        products = product_names(product_entries)

        if not products:
            send_message(vk_session, user_id,
                         "Нейросеть не нашла товары\n\n"
                         "Фотографируй только строки с товарами\n"
                         "Убери тени и блики"
                         )
            return

        receipt_id = f"receipt_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_temp_receipt(user_id, receipt_id, product_entries, amount, receipt_date)

        show_receipt_actions(vk_session, user_id, products, amount, receipt_date)

    except Exception as e:
        logger.error(f"Ошибка обработки фото: {e}")
        send_message(vk_session, user_id, "Ошибка при обработке фото. Попробуй ещё раз.")


def handle_text(vk_session, user_id, text):
    """Обработка текста"""
    current_state = user_states.get(user_id)
    if current_state == 'waiting_for_corrections':
        handle_corrections(vk_session, user_id, text)
        return

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    if not lines:
        return

    send_message(vk_session, user_id, "Отправляю в нейросеть...")

    product_entries = ner_client.predict(lines, filter_service=True)
    products = product_names(product_entries)

    if not products:
        send_message(vk_session, user_id,
                     "Нейросеть не нашла товары\n\n"
                     "Убери служебные строки (ндс, итого, всего)"
                     )
        return

    receipt_date = extract_date_from_ocr(lines)
    amount = extract_amount_from_ocr(lines)

    receipt_id = f"receipt_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_temp_receipt(user_id, receipt_id, product_entries, amount, receipt_date)

    show_receipt_actions(vk_session, user_id, products, amount, receipt_date)


def main():
    logger.info(f"Модель: {MODEL_SERVICE_URL}")

    if not VK_TOKEN:
        raise RuntimeError("VK_TOKEN is not set")
    if not VK_GROUP_ID:
        raise RuntimeError("VK_GROUP_ID is not set")

    if ner_client.check_health():
        logger.info("Модель доступна")
    else:
        logger.warning("Модель не доступна! Запусти model_service.py")

    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract: {version}")
    except Exception as e:
        logger.error(f"Tesseract: {e}")

    # Инициализация VK
    vk_session = vk_api.VkApi(token=VK_TOKEN)

    logger.info("VK Long Poll запускается...")
    longpoll = VkBotLongPoll(vk_session, VK_GROUP_ID)

    logger.info("Бот VK запущен и ожидает сообщения")

    for event in longpoll.listen():
        if event.type == VkBotEventType.MESSAGE_NEW:
            msg = event.object.message
            user_id = msg['from_id']
            text = msg.get('text', '').strip()

            logger.info(f"Получено сообщение: '{text}' от пользователя {user_id}")

            # Обработка команд
            if text in ['Начать', '/start', 'Start']:
                handle_start(vk_session, user_id)
            elif text in ['Статистика за день', '/day', 'День']:
                handle_day(vk_session, user_id)
            elif text in ['Статистика за неделю', '/week', 'Неделя']:
                handle_week(vk_session, user_id)
            elif text in ['Статистика за месяц', '/month', 'Месяц']:
                handle_month(vk_session, user_id)
            elif text in ['Отменить', 'Отменить последний чек', '/cancel', 'Отмена']:
                handle_cancel(vk_session, user_id)
            elif text in ['Сохранить', '/save']:
                handle_save(vk_session, user_id)
            elif text in ['Исправить', '/change']:
                handle_change(vk_session, user_id)
            elif text and not (text.startswith('/')):
                handle_text(vk_session, user_id, text)

            # Обработка вложений (фото)
            if 'attachments' in msg:
                for attachment in msg['attachments']:
                    if attachment['type'] == 'photo':
                        photo_obj = attachment['photo']
                        handle_photo(vk_session, user_id, photo_obj)


if __name__ == "__main__":
    main()
