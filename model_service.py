from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import sys
import socket

MODEL_DIR = "model"
MAX_LEN = 128
HOST = "127.0.0.1"
PORT = 8000
FIX_MISTRAL_REGEX = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NER Model Service for Receipts")


def is_port_in_use(port: int, host: str) -> bool:
    """Проверяет, занят ли порт"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except socket.error:
            return True


class NERRequest(BaseModel):
    lines: List[str]
    filter_service: bool = True


class NERResponse(BaseModel):
    products: List[str]
    processed_lines: int
    model_version: str = "receipt-ner-v1"


try:
    logger.info(f"📁 Загрузка модели из {MODEL_DIR}")

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=True,
        fix_mistral_regex=FIX_MISTRAL_REGEX  # добавляем флаг!
    )

    # Загружаем модель
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()
    label_list = ["O", "B", "I"]
    logger.info("✅ NER модель загружена")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {e}")
    tokenizer, model = None, None


def is_service_line(line: str) -> bool:
    """Проверка на служебную строку"""
    line_lower = line.lower()

    service_words = [
        'ндс',  'налог', 'ставка', 'сумма', 'итог', 'итого', 'всего',
        'касса', 'чек', 'карта', 'наличные', 'безналичные',
        'кассир', 'адрес', 'телефон', 'сайт', 'спасибо',
        'фн', 'фд', 'ккт', 'инн', 'огрн', 'ип', 'ооо',
        'время', 'дата', 'операция', 'покупка', 'возврат',
        'эквайринг', 'терминал', 'сдача', 'внесено',
        'нас', 'hac', 'наc', 'нac', 'не облагается', 'товар'
    ]

    for word in service_words:
        if word in line_lower:
            return True

    digits = sum(c.isdigit() for c in line)
    if digits > len(line) * 0.5:
        return True

    return False


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


def ner_predict_single(text: str) -> List[str]:
    """NER для одной строки"""
    if not text or len(text) < 3:
        return []

    if not tokenizer or not model:
        cleaned = ' '.join(clean_text(text))
        return [cleaned] if cleaned else []

    try:
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
            raw = text[s["start"]:s["end"]].strip()
            tokens = clean_text(raw)
            if tokens:
                product = " ".join(tokens)
                if len(product) > 2:
                    products.append(product)

        if not products:
            tokens = clean_text(text)
            if tokens:
                products.append(" ".join(tokens))

        return products[:3]

    except Exception as e:
        logger.error(f"❌ Ошибка NER: {e}")
        cleaned = ' '.join(clean_text(text))
        return [cleaned] if cleaned else []


@app.get("/")
async def root():
    return {
        "service": "NER Model Service for Receipts",
        "status": "running",
        "model_loaded": tokenizer is not None and model is not None
    }


@app.get("/health")
async def health():
    if tokenizer and model:
        return {"status": "healthy", "model": "loaded"}
    return {"status": "degraded", "model": "not loaded"}


@app.post("/predict", response_model=NERResponse)
async def predict(request: NERRequest):
    if not tokenizer or not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    all_products = []
    processed = 0

    for line in request.lines:
        if request.filter_service and is_service_line(line):
            logger.info(f"Фильтр: {line[:50]}")
            continue

        products = ner_predict_single(line)
        all_products.extend(products)
        processed += 1

        if products:
            logger.info(f"- {line[:50]} -> {products}")

    seen = set()
    unique = []
    for p in all_products:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return NERResponse(
        products=unique,
        processed_lines=processed
    )


if __name__ == "__main__":
    print(f"Проверка порта {PORT}...")

    # Если порт занят, ищем свободный
    if is_port_in_use(PORT, HOST):
        for new_port in range(8001, 8100):
            if not is_port_in_use(new_port, HOST):
                print(f"Порт {PORT} занят, используем {new_port}")
                PORT = new_port
                break

    print(f"Запуск NER сервиса на http://{HOST}:{PORT}")
    print(f"Модель: {MODEL_DIR}")
    print(f"fix_mistral_regex = {FIX_MISTRAL_REGEX}")

    uvicorn.run(app, host=HOST, port=PORT)