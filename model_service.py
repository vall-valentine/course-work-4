from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import socket

from settings import FIX_MISTRAL_REGEX, HOST, MAX_LEN, MODEL_DIR, PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
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


class PredictionItem(BaseModel):
    product: str
    raw_line: str


class NERResponse(BaseModel):
    products: List[str]
    predictions: List[PredictionItem]
    processed_lines: int
    model_version: str = "receipt-ner-v1"


try:
    logger.info(f"📁 Загрузка модели из {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=True,
        fix_mistral_regex=FIX_MISTRAL_REGEX
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR,
        ignore_mismatched_sizes=True
    )
    model.eval()
    label_list = ["O", "B", "I"]
    logger.info("NER модель загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    tokenizer, model = None, None


def split_by_service_keyword(line: str) -> str | None:
    """
    Обрезает строку по первому служебному слову (НДС, итого, налог и т.д.).
    Возвращает только левую часть до слова, без самого слова и всего, что справа.
    Если слово в начале строки — возвращает None.
    """
    # Ключевые слова, после которых всё отсекается
    service_keywords = [
        "\u043d\u0434\u0441",        # ндс
        "\u043d\u0430\u043b\u043e\u0433",      # налог
        "\u0438\u0442\u043e\u0433",       # итог
        "\u0438\u0442\u043e\u0433\u043e",      # итого
        "\u0432\u0441\u0435\u0433\u043e",      # всего
        "\u0441\u0443\u043c\u043c\u0430",      # сумма
        "\u0441\u043a\u0438\u0434\u043a\u0430",     # скидка
        "\u0434\u0438\u0441\u043a\u043e\u043d\u0442",    # дисконт
        "\u0430\u043a\u0446\u0438\u044f",      # акция
        "\u043a\u0430\u0441\u0441\u0438\u0440",     # кассир
        "\u043e\u043f\u043b\u0430\u0442\u0430",     # оплата
        "\u043d\u0430\u043b\u0438\u0447\u043d\u044b\u0435",   # наличные
        "\u0431\u0435\u0437\u043d\u0430\u043b",     # безнал
        "\u043a\u0430\u0440\u0442\u0430",      # карта
    ]

    line_lower = line.lower()
    best_cut_position = None

    for keyword in service_keywords:
        # Ищем keyword как отдельное слово
        pattern = rf"\b{re.escape(keyword)}\b"
        match = re.search(pattern, line_lower)
        if match:
            if best_cut_position is None or match.start() < best_cut_position:
                best_cut_position = match.start()

    if best_cut_position is not None:
        if best_cut_position == 0:
            return None
        left_part = line[:best_cut_position].strip()
        return left_part if left_part else None

    # Если служебных слов нет, возвращаем строку как есть
    return line


def is_fully_service_line(line: str) -> bool:
    """Проверка на полностью служебную строку (много цифр или короткая)"""
    if not line or len(line) < 3:
        return True

    # Если больше половины символов — цифры
    digits = sum(c.isdigit() for c in line)
    if digits > len(line) * 0.5:
        return True

    return False


def is_service_line(line: str) -> bool:
    """
    Backward-compatible helper: detect lines that should be filtered out entirely.

    This matches the current pipeline logic:
    - if a service keyword appears at the start -> service line
    - if a service keyword appears later but the remaining "left part" is still not a product
      (too short / too many digits) -> service line
    """
    if line is None:
        return False
    original = line.strip()
    if not original:
        return False

    # Purely numeric garbage lines aren't "service"; we just ignore them elsewhere.
    if original.isdigit():
        return False

    # Explicit service lines: keyword at the beginning, or the whole line is a service line after trimming.
    trimmed = split_by_service_keyword(original)
    if trimmed is None:
        return True
    return is_fully_service_line(trimmed)




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
    text = re.sub(r'["\'\/@°€#$%=]+', ' ', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    # Токенизация
    tokens = text.split()
    # Удаляем одиночные символы и цифры
    tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]
    # Удаляем стоп-слова
    if stopwords_set is not None:
        tokens = [t for t in tokens if t not in stopwords_set]
    return " ".join(tokens)


def clean_text_tokens(text: str) -> List[str]:
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []


def ner_predict_single(text: str) -> List[str]:
    """NER для одной строки"""
    if not text or len(text) < 3:
        return []

    if not tokenizer or not model:
        cleaned = clean_text(text)
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
            product = clean_text(raw)
            if product and len(product) > 2:
                products.append(product)

        if not products:
            product = clean_text(text)
            if product:
                products.append(product)

        return products[:3]

    except Exception as e:
        logger.error(f"Ошибка NER: {e}")
        cleaned = clean_text(text)
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

    all_predictions = []
    processed = 0

    for line in request.lines:
        model_input = line

        if request.filter_service:
            # Обрезаем по служебному слову
            trimmed = split_by_service_keyword(line)
            if not trimmed or is_fully_service_line(trimmed):
                logger.info(f"Фильтр (обрезано по НДС/итого): {line[:50]}")
                continue
            model_input = trimmed

        products = ner_predict_single(model_input)
        all_predictions.extend(
            {"product": product, "raw_line": line} for product in products
        )
        processed += 1

        if products:
            logger.info(f"- Исходная: {line[:50]} | После обрезки: {model_input[:40]} -> {products}")

    seen = set()
    unique = []
    unique_predictions = []
    for prediction in all_predictions:
        product = prediction["product"]
        if product not in seen:
            seen.add(product)
            unique.append(product)
            unique_predictions.append(prediction)

    return NERResponse(
        products=unique,
        predictions=unique_predictions,
        processed_lines=processed
    )


if __name__ == "__main__":
    print(f"Проверка порта {PORT}...")

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
