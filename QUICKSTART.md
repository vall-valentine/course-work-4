# Quick Start Guide

## Быстрый запуск проекта

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### 2. Настройка
```bash
# Скопируйте .env.example в .env
copy .env.example .env

# Отредактируйте .env, укажите ваши токены
notepad .env
```

### 3. Тесты (опционально)
```bash
pytest tests/test_ner.py -v
```

### 4. Запуск локально

**Шаг 1: ML сервис**
```bash
python model_service.py
```

**Шаг 2: VK бот** (в новом терминале)
```bash
python vk_bot.py
```

### 5. Автоматическая разметка (когда накопите данные)
```bash
# Проверить данные
python auto_labeling.py --min-samples 50

# Разметить всё
python auto_labeling.py
```

### 6. Дообучение модели (после разметки)
```bash
# Проверить
python train_finetune.py --dry-run

# Дообучить
python train_finetune.py --min-samples 100

# Кастомные параметры
python train_finetune.py --learning-rate 3e-5 --epochs 5
```

### 7. Просмотр результатов MLflow
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:./mlruns
```
Откроется на `http://localhost:5000`

---

## Docker (альтернатива локальному запуску)

### Запуск
```bash
docker-compose up -d
```

### Логи
```bash
docker-compose logs -f ml-service
docker-compose logs -f vk-bot
```

### MLflow UI
Открыть `http://localhost:5000`

### Остановка
```bash
docker-compose down
```
