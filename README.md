# VK Бот для извлечения товаров из чеков с NER-моделью

Автоматизированная система для распознавания чеков, извлечения названий товаров с помощью нейросети (NER) и ведения истории покупок.

## Возможности

### Основная функциональность
- **Распознавание чеков** через фото (OCR с Tesseract)
- **Прием текста** чеков напрямую
- **NER-модель** для извлечения названий товаров
- **Автоматическое выделение суммы** из чека (ИТОГ/ИТОГО с учетом опечаток OCR)
- **Автоматическое выделение даты** из чека (формат ДД.ММ.ГГГГ)
- **Сохранение истории** покупок в SQLite
- **Статистика** покупок за день/неделю/месяц

---

## Система распознавания товаров из чеков состоит из:

```
Фото чека -> OCR (Tesseract) -> Строки текста -> Фильтр слов -> NER модель -> Товары
```

**Компоненты:**
1. **OCR модуль** (Tesseract) - распознает текст с фото чека
2. **Фильтрация** - отбор служебных строк (ИТОГО, НДС и т.д.)
3. **NER модель** (Token Classification) - извлекает названия товаров из строк
4. **Постобработка** - очистка и нормализация названий товаров

**Тестирование:** `pytest tests/test_ner.py -v`

## Автоматическая разметка данных

**Автоматическая фильтрация** служебных строк по ключевым словам

**Эвристические правила** для определения позиций товаров в строках

**Подготовка обучающих данных:**
- Сохранение `raw_line` и `product` в БД
- Конвертация в BIO-теги (Begin, Inside, Outside)
- Экспорт в JSON, CoNLL и training форматах

**Запуск:**
```bash
python auto_labeling.py --min-samples 100
```

## Дообучение и тонкая настройка оптимальной модели

**Процесс дообучения:**
1. Загрузка размеченных данных из `auto_labeling.py`
2. Токенизация и создание BIO-тегов
3. Fine-tuning с параметрами (lr=2e-5, epochs=5, batch_size=16)
4. Оценка на validation set (precision, recall, F1)
5. Сохранение лучшей модели в `model_finetuned/`

**MLflow трекинг:**
- Все эксперименты логируются в MLflow
- Метрики: train/eval loss, precision, recall, F1-score
- Параметры: learning rate, epochs, batch size, dataset size
- Артефакты: сохранённые модели, metadata

---

## Компоненты системы

### 1. VK Бот (`vk_bot.py`)
- Прием фото и сообщений от пользователей ВКонтакте
- Интеграция с OCR (Tesseract)
- Взаимодействие с ML-сервисом через HTTP API
- **Выделение даты** из чека регуляркой `\d{2}\.\d{2}\.\d{4}`
- **Выделение суммы** из чека (ИТОГ/ИТОГО с учетом опечаток OCR)
- Показ распознанных товаров пользователю

### 2. ML Сервис (`model_service.py`)
- FastAPI сервер для NER
- NER модель для извлечения товаров
- Фильтрация служебных строк
- API для дообучения модели

### 3. Автоматическая разметка (`auto_labeling.py`)
- Загрузка данных из БД
- Создание BIO-тегов
- Экспорт в JSON/CoNLL/training форматы
- Статистика по размеченным данным

### 4. Дообучение модели (`train_finetune.py`)
- Fine-tuning NER модели
- MLflow трекинг экспериментов
- Оценка качества (precision, recall, F1)
- Сохранение улучшенной модели

---

## Команды бота

| Команда (RU) | Команда (EN) | Описание |
|--------------|--------------|----------|
| Начать | /start | Начать работу с ботом |
| День | /day | Статистика покупок за день |
| Неделя | /week | Статистика покупок за неделю |
| Месяц | /month | Статистика покупок за месяц |
| Сохранить | /save | Сохранить распознанный чек |
| Исправить | /change | Исправить список товаров |
| Отменить | /cancel | Отменить последний чек |

---

## Установка и запуск

### Вариант 1: Локальный запуск

#### Требования
- Python 3.9+
- Tesseract OCR
- VK Group Token
- Обученная NER модель

#### Установка зависимостей
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt  # для тестов
```

#### Установка Tesseract OCR
**Windows:**
1. Скачать https://github.com/UB-Mannheim/tesseract/wiki
2. Установить в `C:\Program Files\Tesseract-OCR`
3. Указать путь в `vk_bot.py`

**Linux:**
```bash
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-rus
```

#### Запуск системы

**Шаг 1: ML сервис**
```bash
python model_service.py
```

**Шаг 2: VK бот**
```bash
python vk_bot.py
```

### Вариант 2: Docker (рекомендуется)

#### Запуск всех сервисов
```bash
docker-compose up -d
```

#### Логи
```bash
docker-compose logs -f ml-service
docker-compose logs -f vk-bot
docker-compose logs -f mlflow
```

#### Остановка
```bash
docker-compose down
```

#### MLflow UI (в Docker)
Открыть `http://localhost:5000`

---

## API Endpoints ML сервиса

### Основные
- `GET /` - проверка статуса сервиса
- `GET /health` - health check
- `POST /predict` - NER для извлечения товаров

### Примеры использования API

**Извлечение товаров:**
```python
import requests

response = requests.post("http://127.0.0.1:8000/predict", json={
    "lines": [
        "Молоко 3.2% 1л",
        "Хлеб бородинский 400г",
        "ИТОГО: 250.00"
    ],
    "filter_service": True
})

print(response.json())
```

---

## Автоматическая разметка

### Запуск
```bash
# Разметить все данные
python auto_labeling.py

# Только последние 7 дней
python auto_labeling.py --last-days 7

# С минимальным количеством
python auto_labeling.py --min-samples 200
```

### Экспортированные файлы
- `exported_data/bio_labeled_data_*.json` - JSON с BIO-тегами
- `exported_data/bio_labeled_data_*.conll` - CoNLL-2003 формат
- `exported_data/training_data_*.json` - формат для transformers

---

## Дообучение модели

### Запуск
```bash
# Проверить данные
python train_finetune.py --dry-run

# Дообучить
python train_finetune.py --min-samples 100

# Кастомные параметры
python train_finetune.py --learning-rate 3e-5 --epochs 5 --batch-size 32
```

### Результаты
- **Консоль**: метрики сразу после обучения
- **MLflow UI**: `mlflow ui` → `http://localhost:5000`
- **Файл**: `model_finetuned/metrics_report.json`

### Метрики
- Precision/Recall/F1 (weighted average)
- Entity-level Precision/Recall/F1 (только B и I теги)
- Accuracy
- Train/Eval Loss

---

## Тестирование

### Запуск тестов
```bash
pytest tests/test_ner.py -v
```

### Покрытие
- Фильтрация служебных строк
- Очистка текста
- BIO-разметка
- NER предсказание
- Интеграционный pipeline

---

## Структура проекта

```
диплом/
├── vk_bot.py                     # VK бот с OCR
├── model_service.py              # FastAPI ML-сервис (NER)
├── auto_labeling.py              # Автоматическая разметка
├── train_finetune.py             # Дообучение модели с MLflow
├── requirements.txt              # Основные зависимости
├── requirements-test.txt         # Зависимости для тестов
├── .env.example                  # Пример настроек
├── Dockerfile.ml                 # Docker для ML-сервиса
├── Dockerfile.bot                # Docker для VK-бота
├── docker-compose.yml            # Оркестрация сервисов
├── .dockerignore                 # Исключения для Docker
│
├── tests/                        # Тесты
│   └── test_ner.py
│
├── model/                        # Базовая NER модель
├── model_finetuned/              # Дообученная модель
├── exported_data/                # Размеченные данные
│   ├── bio_labeled_data.json
│   ├── bio_labeled_data.conll
│   └── training_data.json
│
├── mlruns/                       # MLflow артефакты
├── data/                         # SQLite БД (Docker)
│   └── purchases.db
└── debug_images/                 # Отладочные изображения
```

---

## Структура базы данных

### Таблица `purchases`
- `id` - уникальный ID
- `user_id` - ID пользователя VK
- `product` - название товара (распознано NER)
- `raw_line` - исходная строка из чека
- `receipt_id` - ID чека
- `amount` - сумма чека (руб.)
- `receipt_date` - дата чека (ДД.ММ.ГГГГ)
- `category` - категория товара
- `ts` - временная метка сохранения

### Таблица `temp_receipts`
- Временное хранение распознанных чеков
- Используется до подтверждения пользователем

---

## Возможные проблемы

### Модель не найдена
- Проверьте что папка `model/` существует
- Проверьте логи `model_service.py`

### OCR не распознает текст
- Убедитесь что Tesseract установлен
- Проверьте путь в `vk_bot.py`

### Недостаточно данных для дообучения
- Накопите больше чеков
- Уменьшите `--min-samples`
- Увеличьте `--last-days`

### Порт 8000 занят
- ML сервис автоматически найдет свободный порт (8001-8100)

### Docker проблемы
```bash
# Пересобрать образы
docker-compose build --no-cache

# Очистить volumes
docker-compose down -v

# Проверить логи
docker-compose logs -f
```


