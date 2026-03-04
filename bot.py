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
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, ContentType
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from concurrent.futures import ThreadPoolExecutor
import functools

TOKEN = os.getenv("BOT_TOKEN")
DB_PATH = "purchases.db"
DEBUG_FOLDER = "debug_images"
MODEL_HOST = "127.0.0.1"
MODEL_PORT = 8000

os.makedirs(DEBUG_FOLDER, exist_ok=True)

bot = Bot(token=TOKEN)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()
executor = ThreadPoolExecutor(max_workers=2)


def find_model_service():
    """Ищет запущенный сервер модели на разных портах"""
    for port in range(8000, 8010):
        try:
            url = f"http://{MODEL_HOST}:{port}/health"
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                logger.info(f"✅ Модель найдена на порту {port}")
                return port
        except:
            continue
    logger.warning("⚠️ Модель не найдена")
    return None


MODEL_PORT = find_model_service() or 8000
MODEL_SERVICE_URL = f"http://{MODEL_HOST}:{MODEL_PORT}"


class ReceiptStates(StatesGroup):
    waiting_for_corrections = State()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS purchases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            product TEXT,
            raw_line TEXT,
            receipt_id TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS temp_receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            receipt_id TEXT,
            products TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()


def save_temp_receipt(user_id: int, receipt_id: str, products: list):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM temp_receipts WHERE user_id = ?", (str(user_id),))
        cur.execute(
            "INSERT INTO temp_receipts (user_id, receipt_id, products) VALUES (?, ?, ?)",
            (str(user_id), receipt_id, ','.join(products))
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
            "SELECT receipt_id, products FROM temp_receipts WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (str(user_id),)
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0], row[1].split(',')
        return None, []
    except Exception as e:
        logger.error(f"Ошибка получения временного чека: {e}")
        return None, []


def delete_temp_receipt(user_id: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM temp_receipts WHERE user_id = ?", (str(user_id),))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка удаления временного чека: {e}")


def save_final_receipt(user_id: int, receipt_id: str, products: list):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        for product in products:
            cur.execute(
                "INSERT INTO purchases (user_id, product, receipt_id) VALUES (?, ?, ?)",
                (str(user_id), product, receipt_id)
            )
        conn.commit()
        conn.close()
        logger.info(f"Сохранен чек {receipt_id} с {len(products)} товарами")
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


def get_stats(user_id: int, days: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cutoff = datetime.now() - timedelta(days=days)
    cur.execute(
        "SELECT product, COUNT(*) FROM purchases WHERE user_id = ? AND ts > ? GROUP BY product ORDER BY COUNT(*) DESC",
        (str(user_id), cutoff))
    rows = cur.fetchall()
    conn.close()
    return rows


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
                return data.get("products", [])
            else:
                logger.error(f"Ошибка модели: {response.status_code}")
                return []

        except requests.exceptions.ConnectionError:
            logger.error("Модель не доступна!")
            return []
        except Exception as e:
            logger.error(f"Ошибка запроса к модели: {e}")
            return []


# Создаем клиента
ner_client = NERClient(MODEL_SERVICE_URL)


def ocr_image_sync(image_bytes: bytes, user_id: int) -> list:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return []

        height, width = img.shape[:2]
        if height < 800:
            scale = 1200 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # debug_path = f"{DEBUG_FOLDER}/user_{user_id}_{timestamp}.jpg"
        # cv2.imwrite(debug_path, binary)

        pil_img = Image.fromarray(binary)
        text = pytesseract.image_to_string(pil_img, lang='rus+eng', config='--oem 3 --psm 6')

        raw_lines = [line.strip() for line in text.split('\n') if line.strip()]

        return raw_lines

    except Exception as e:
        logger.error(f"Ошибка OCR: {e}")
        return []


async def ocr_image(image_bytes: bytes, user_id: int) -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        functools.partial(ocr_image_sync, image_bytes, user_id)
    )


@router.message(Command("start"))
async def cmd_start(msg: Message):
    text = (
        "👋 Привет! Я помогаю сохранять товары из чеков\n\n"
        "📸 Отправь фото строк с товарами\n"
        "📝 Или вставь текст\n\n"
        "После распознавания:\n"
        "/save - сохранить\n"
        "/change - исправить\n"
        "/cancel - отменить последний\n\n"
        "📊 Статистика:\n"
        "/day /week /month"
    )
    await msg.answer(text)


@router.message(Command("day"))
async def cmd_day(msg: Message):
    rows = get_stats(msg.from_user.id, 1)
    if not rows:
        await msg.answer("📭 За сегодня покупок нет")
        return

    total = sum(c for _, c in rows)
    lines = [f"📊 За сегодня (всего покупок: {total})", ""]
    for prod, cnt in rows[:10]:
        lines.append(f"• {prod}: {cnt}")
    await msg.answer("\n".join(lines))


@router.message(Command("week"))
async def cmd_week(msg: Message):
    rows = get_stats(msg.from_user.id, 7)
    if not rows:
        await msg.answer("📭 За неделю покупок нет")
        return

    total = sum(c for _, c in rows)
    lines = [f"📊 За неделю (всего покупок: {total})", ""]
    for prod, cnt in rows[:10]:
        lines.append(f"• {prod}: {cnt}")
    await msg.answer("\n".join(lines))


@router.message(Command("month"))
async def cmd_month(msg: Message):
    rows = get_stats(msg.from_user.id, 30)
    if not rows:
        await msg.answer("📭 За месяц покупок нет")
        return

    total = sum(c for _, c in rows)
    lines = [f"📊 За месяц (всего покупок: {total})", ""]
    for prod, cnt in rows[:15]:
        lines.append(f"• {prod}: {cnt}")
    await msg.answer("\n".join(lines))


@router.message(Command("cancel"))
async def cmd_cancel(msg: Message):
    deleted = delete_last_receipt(msg.from_user.id)
    if deleted > 0:
        await msg.answer(f"Последний чек удален (удалено {deleted} записей)")
    else:
        await msg.answer("Нет чека для удаления")


@router.message(Command("save"))
async def cmd_save(msg: Message, state: FSMContext):
    receipt_id, products = get_temp_receipt(msg.from_user.id)

    if not products:
        await msg.answer("Нет чека для сохранения")
        return

    save_final_receipt(msg.from_user.id, receipt_id, products)
    delete_temp_receipt(msg.from_user.id)
    await state.clear()

    lines = [f"Чек сохранен! ({len(products)} товаров)", ""]
    for i, p in enumerate(products[:10], 1):
        lines.append(f"{i}. {p}")

    await msg.answer("\n".join(lines))


@router.message(Command("change"))
async def cmd_change(msg: Message, state: FSMContext):
    receipt_id, products = get_temp_receipt(msg.from_user.id)

    if not products:
        await msg.answer("Нет чека для редактирования")
        return

    lines = ["Текущий список товаров:", ""]
    for i, p in enumerate(products, 1):
        lines.append(f"{i}. {p}")

    lines.append("")
    lines.append("Отправьте исправления:")
    lines.append("1 молоко 3.2%")
    lines.append("3 хлеб бородинский")
    lines.append("")
    lines.append("/save - оставить как есть")
    lines.append("/cancel - отменить")

    await msg.answer("\n".join(lines))

    await state.set_state(ReceiptStates.waiting_for_corrections)
    await state.update_data(receipt_id=receipt_id, products=products)


@router.message(ReceiptStates.waiting_for_corrections, F.text)
async def process_corrections(msg: Message, state: FSMContext):
    data = await state.get_data()
    products = data.get('products', [])
    receipt_id = data.get('receipt_id')

    if not products:
        await msg.answer("Ошибка: чек не найден")
        await state.clear()
        return

    lines = msg.text.strip().split('\n')
    corrections = []
    indices_to_delete = []  # для отслеживания удаляемых позиций

    # Сначала собираем все исправления
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Формат: "номер товар" или "номер" (пустота)
        match = re.match(r'^(\d+)(?:\s+(.*))?$', line)
        if match:
            num = int(match.group(1))
            new_product = match.group(2)  # может быть None если пусто

            if 1 <= num <= len(products):
                if new_product is None or new_product.strip() == "":
                    # Пустота - значит удалить товар
                    indices_to_delete.append(num)
                    logger.info(f"🗑️ Будет удален товар #{num}: {products[num - 1]}")
                else:
                    # Есть текст - исправляем
                    corrections.append((num, new_product.strip().lower()))
            else:
                await msg.answer(f"Номер {num} вне диапазона (1-{len(products)})")
                return
        else:
            await msg.answer(f"Неправильный формат: {line}\nНужно: номер товар или просто номер для удаления")
            return

    # Применяем исправления (сначала исправляем, потом удаляем)
    for num, new_product in corrections:
        products[num - 1] = new_product
        logger.info(f"Исправлен #{num}: {new_product}")

    # Удаляем товары (с конца, чтобы не сбить индексы)
    for num in sorted(indices_to_delete, reverse=True):
        deleted = products.pop(num - 1)
        logger.info(f"🗑️ Удален товар #{num}: {deleted}")

    # Сохраняем обновленный список
    save_temp_receipt(msg.from_user.id, receipt_id, products)

    # Показываем результат
    if not products:
        # Если все товары удалили
        delete_temp_receipt(msg.from_user.id)
        await state.clear()
        await msg.answer("Все товары удалены. Чек отменен.")
        return

    result = ["Список обновлен:", ""]
    for i, p in enumerate(products, 1):
        result.append(f"{i}. {p}")

    result.append("")
    result.append("/save - сохранить")
    result.append("/change - продолжить правку")
    result.append("/cancel - отменить")

    await msg.answer("\n".join(result))


@router.message(F.content_type == ContentType.PHOTO)
async def process_photo(msg: Message, state: FSMContext):
    user_id = msg.from_user.id

    current_state = await state.get_state()
    if current_state == ReceiptStates.waiting_for_corrections.state:
        await msg.answer("⚠️ Сначала завершите редактирование (/save или /cancel)")
        return

    status = await msg.answer("Распознаю текст...")

    try:
        photo = msg.photo[-2] if len(msg.photo) > 1 else msg.photo[-1]
        file = await bot.get_file(photo.file_id)
        image_bytes = await bot.download_file(file.file_path)

        lines = await asyncio.wait_for(
            ocr_image(image_bytes.read(), user_id),
            timeout=15.0
        )

        if not lines:
            await status.edit_text("Не удалось распознать текст")
            return

        await status.edit_text(f"Распознано {len(lines)} строк, отправляю в нейросеть...")

        products = ner_client.predict(lines, filter_service=True)

        if not products:
            await status.edit_text(
                "Нейросеть не нашла товары\n\n"
                "• Фотографируй только строки с товарами\n"
                "• Убери тени и блики, например вспышкой"
            )
            return

        receipt_id = f"receipt_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_temp_receipt(user_id, receipt_id, products)

        result = [f"🔍 Найдено {len(products)} товаров:", ""]
        for i, p in enumerate(products, 1):
            result.append(f"{i}. {p}")

        result.append("")
        result.append("✅ /save - всё верно")
        result.append("✏️ /change - исправить")
        result.append("❌ /cancel - отменить")

        await status.edit_text("\n".join(result))

    except asyncio.TimeoutError:
        await status.edit_text("Превышено время обработки")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await status.edit_text("Ошибка. Попробуйте ещё раз.")


@router.message(F.text)
async def process_text(msg: Message, state: FSMContext):
    user_id = msg.from_user.id

    if msg.text.startswith('/'):
        return

    current_state = await state.get_state()
    if current_state == ReceiptStates.waiting_for_corrections.state:
        return

    lines = [line.strip() for line in msg.text.split('\n') if line.strip()]

    if not lines:
        return

    status = await msg.answer("Отправляю в нейросеть...")

    products = ner_client.predict(lines, filter_service=True)

    if not products:
        await status.edit_text(
            "Нейросеть не нашла товары\n\n"
            "Уберите служебные строки (нас, ндс, итого)"
        )
        return

    receipt_id = f"receipt_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_temp_receipt(user_id, receipt_id, products)

    result = [f"🔍 Найдено {len(products)} товаров:", ""]
    for i, p in enumerate(products, 1):
        result.append(f"{i}. {p}")

    result.append("")
    result.append("✅ /save - всё верно")
    result.append("✏️ /change - исправить")
    result.append("❌ /cancel - отменить")

    await status.edit_text("\n".join(result))


async def main():
    logger.info(f"Отладка: {os.path.abspath(DEBUG_FOLDER)}")
    logger.info(f"Модель: {MODEL_SERVICE_URL}")

    if ner_client.check_health():
        logger.info("Модель доступна")
    else:
        logger.warning("Модель не доступна! Запусти model_service.py")

    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract: {version}")
    except Exception as e:
        logger.error(f"Tesseract: {e}")

    dp.include_router(router)
    logger.info("Бот с FastAPI клиентом запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())