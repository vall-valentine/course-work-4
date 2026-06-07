"""
Автоматическая разметка данных для дообучения NER модели

Загружает сырые данные из БД (raw_line + product) и создаёт BIO-теги:
- B (Begin) - начало названия товара
- I (Inside) - внутри названия товара
- O (Outside) - вне названия товара

Экспортирует в JSON для последующего обучения.
"""

import sqlite3
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from settings import DB_PATH, EXPORT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BIO_LABELS = ["O", "B", "I"]


def init_export_dir():
    """Создаёт директорию для экспорта"""
    Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)


def load_receipt_data(db_path: str = DB_PATH, min_samples: int = 50,
                    last_days: Optional[int] = None) -> List[Dict]:
    """
    Загружает данные из БД для разметки

    Args:
        db_path: путь к БД
        min_samples: минимальное количество записей
        last_days: загрузить только за последние N дней (None = все)

    Returns:
        Список словарей {raw_line, product, receipt_id, ts}
    """
    if not Path(db_path).exists():
        logger.error(f"База данных не найдена: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        # Проверяем, есть ли нужные колонки
        cur.execute("PRAGMA table_info(purchases)")
        columns = [row['name'] for row in cur.fetchall()]

        if 'raw_line' not in columns or 'product' not in columns:
            logger.error("В таблице purchases нет колонок raw_line или product")
            return []

        # Формируем запрос - используем product как raw_line если raw_line отсутствует
        query = """
            SELECT 
                COALESCE(raw_line, product) as raw_line, 
                product, 
                receipt_id, 
                ts 
            FROM purchases 
            WHERE product IS NOT NULL
        """
        params = []

        if last_days:
            query += " AND ts > datetime('now', ?)"
            params.append(f"-{last_days} days")

        query += " ORDER BY ts DESC"

        cur.execute(query, params)
        rows = cur.fetchall()

        # Конвертируем в список словарей
        data = []
        for row in rows:
            data.append({
                'raw_line': row['raw_line'],
                'product': row['product'],
                'receipt_id': row['receipt_id'],
                'ts': row['ts']
            })

        logger.info(f"Загружено {len(data)} записей из БД")

        if len(data) < min_samples:
            logger.warning(f"⚠️ Данных мало: {len(data)} < {min_samples}")
            logger.warning("Рекомендуется накопить больше данных для качественного дообучения")

        return data

    except Exception as e:
        logger.error(f"Ошибка загрузки из БД: {e}")
        return []
    finally:
        conn.close()


def find_product_position(raw_line: str, product: str) -> Optional[Tuple[int, int]]:
    """
    Находит позицию товара в исходной строке

    Args:
        raw_line: исходная строка из чека
        product: распознанный товар

    Returns:
        (start_pos, end_pos) или None если не найдено
    """
    if not raw_line or not product:
        return None

    # Нормализуем для поиска
    line_lower = raw_line.lower().strip()
    product_lower = product.lower().strip()

    # Убираем лишние пробелы в product
    product_normalized = re.sub(r'\s+', ' ', product_lower)

    # Пробуем точное совпадение
    start = line_lower.find(product_normalized)
    if start != -1:
        return (start, start + len(product_normalized))

    # Пробуем посимвольное нечёткое совпадение
    # (OCR мог добавить ошибки)
    product_words = product_normalized.split()
    if len(product_words) > 1:
        # Ищем по ключевым словам (минимум 2 слова)
        for i in range(len(product_words) - 1):
            phrase = f"{product_words[i]} {product_words[i+1]}"
            start = line_lower.find(phrase)
            if start != -1:
                # Нашли часть фразы, пробуем расширить
                return (start, start + len(phrase))

    # Если не нашли точно, пробуем по первому слову
    first_word = product_words[0] if product_words else ""
    if len(first_word) > 3:  # только если слово достаточно длинное
        start = line_lower.find(first_word)
        if start != -1:
            # Приблизительная позиция
            return (start, start + len(first_word))

    return None


def create_bio_tags(raw_line: str, product: str) -> Optional[List[Tuple[str, str]]]:
    """
    Создаёт BIO-теги для токенов строки

    Args:
        raw_line: исходная строка из чека
        product: распознанный товар

    Returns:
        Список кортежей (token, bio_tag) или None если ошибка
    """
    if not raw_line or not product:
        return None

    # Находим позицию товара
    position = find_product_position(raw_line, product)
    if not position:
        logger.debug(f"Не найдена позиция товара '{product}' в строке '{raw_line}'")
        return None

    prod_start, prod_end = position

    # Токенизируем строку (по пробелам с сохранением позиций)
    tokens = []
    for match in re.finditer(r'\S+', raw_line):
        token = match.group()
        token_start = match.start()
        token_end = match.end()
        tokens.append({
            'text': token,
            'start': token_start,
            'end': token_end
        })

    if not tokens:
        return None

    # Назначаем BIO-теги
    bio_tags = []
    for token_info in tokens:
        t_start = token_info['start']
        t_end = token_info['end']

        # Проверяем пересечение с позицией товара
        if t_start >= prod_start and t_end <= prod_end:
            # Токен внутри товара
            if t_start == prod_start or (len(bio_tags) > 0 and bio_tags[-1][1] == 'O'):
                bio_tags.append((token_info['text'], 'B'))
            else:
                bio_tags.append((token_info['text'], 'I'))
        elif t_start < prod_end and t_end > prod_start:
            # Частичное пересечение
            if t_start >= prod_start:
                bio_tags.append((token_info['text'], 'B'))
            else:
                bio_tags.append((token_info['text'], 'I'))
        else:
            # Вне товара
            bio_tags.append((token_info['text'], 'O'))

    return bio_tags


def create_bio_sequences(data: List[Dict]) -> List[Dict]:
    """
    Создаёт BIO-последовательности для всех записей

    Args:
        data: список записей из БД

    Returns:
        Список BIO-последовательностей
    """
    bio_sequences = []
    skipped = 0

    for record in data:
        raw_line = record['raw_line']
        product = record['product']

        bio_tags = create_bio_tags(raw_line, product)
        if bio_tags:
            bio_sequences.append({
                'tokens': [tag[0] for tag in bio_tags],
                'ner_tags': [tag[1] for tag in bio_tags],
                'receipt_id': record['receipt_id'],
                'timestamp': record['ts']
            })
        else:
            skipped += 1
            logger.debug(f"Пропущено: product='{product}', line='{raw_line[:50]}'")

    logger.info(f"✅ Создано {len(bio_sequences)} BIO-последовательностей")
    logger.info(f"⚠️ Пропущено {skipped} записей (не найдена позиция товара)")

    return bio_sequences


def export_bio_json(bio_sequences: List[Dict], output_path: str = None) -> str:
    """
    Экспортирует BIO-данные в JSON

    Args:
        bio_sequences: BIO-последовательности
        output_path: путь для сохранения

    Returns:
        путь к файлу
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{EXPORT_DIR}/bio_labeled_data_{timestamp}.json"

    init_export_dir()

    export_data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_sequences': len(bio_sequences),
            'label_scheme': 'BIO',
            'labels': BIO_LABELS,
            'description': 'Автоматически размеченные данные для дообучения NER модели'
        },
        'sequences': bio_sequences
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    logger.info(f"📁 Экспортировано в {output_path}")
    return output_path


def export_conll_format(bio_sequences: List[Dict], output_path: str = None) -> str:
    """
    Экспортирует в формат CoNLL-2003 (стандарт для NER)

    Args:
        bio_sequences: BIO-последовательности
        output_path: путь для сохранения

    Returns:
        путь к файлу
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{EXPORT_DIR}/bio_labeled_data_{timestamp}.conll"

    init_export_dir()

    with open(output_path, 'w', encoding='utf-8') as f:
        for seq in bio_sequences:
            for token, tag in zip(seq['tokens'], seq['ner_tags']):
                f.write(f"{token} {tag}\n")
            f.write("\n")  # пустая строка между последовательностями

    logger.info(f"📁 Экспортировано в CoNLL формат: {output_path}")
    return output_path


def export_training_format(bio_sequences: List[Dict], output_path: str = None) -> str:
    """
    Экспортирует в формат для transformers datasets

    Args:
        bio_sequences: BIO-последовательности
        output_path: путь для сохранения

    Returns:
        путь к файлу
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{EXPORT_DIR}/training_data_{timestamp}.json"

    init_export_dir()

    # Создаём label2id и id2label
    label2id = {label: idx for idx, label in enumerate(BIO_LABELS)}
    id2label = {idx: label for label, idx in label2id.items()}

    training_data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_sequences': len(bio_sequences),
            'label2id': label2id,
            'id2label': id2label
        },
        'data': []
    }

    for seq in bio_sequences:
        # Конвертируем теги в ID
        tag_ids = [label2id[tag] for tag in seq['ner_tags']]

        training_data['data'].append({
            'tokens': seq['tokens'],
            'ner_tags': seq['ner_tags'],
            'ner_tag_ids': tag_ids
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    logger.info(f"📁 Экспортировано в формате для обучения: {output_path}")
    return output_path


def print_statistics(data: List[Dict], bio_sequences: List[Dict]):
    """Печатает статистику по размеченным данным"""
    print("\n" + "="*60)
    print("📊 СТАТИСТИКА АВТОМАТИЧЕСКОЙ РАЗМЕТКИ")
    print("="*60)

    print(f"\n📝 Всего записей в БД: {len(data)}")
    print(f"✅ Размечено последовательностей: {len(bio_sequences)}")
    print(f"⚠️ Пропущено: {len(data) - len(bio_sequences)}")

    if bio_sequences:
        # Считаем токены по классам
        total_tokens = 0
        b_count = 0
        i_count = 0
        o_count = 0

        for seq in bio_sequences:
            tags = seq['ner_tags']
            total_tokens += len(tags)
            b_count += tags.count('B')
            i_count += tags.count('I')
            o_count += tags.count('O')

        print(f"\n🏷️ Распределение BIO-тегов:")
        print(f"  B (Begin):     {b_count:6d} ({b_count/total_tokens*100:.1f}%)")
        print(f"  I (Inside):    {i_count:6d} ({i_count/total_tokens*100:.1f}%)")
        print(f"  O (Outside):   {o_count:6d} ({o_count/total_tokens*100:.1f}%)")
        print(f"  Всего токенов: {total_tokens:6d}")

        # Средняя длина последовательности
        avg_len = sum(len(seq['tokens']) for seq in bio_sequences) / len(bio_sequences)
        print(f"\n📏 Средняя длина строки: {avg_len:.1f} токенов")

        # Уникальных токенов
        all_tokens = set()
        for seq in bio_sequences:
            all_tokens.update(seq['tokens'])
        print(f"🔤 Уникальных токенов: {len(all_tokens)}")

    print("="*60 + "\n")


def run_auto_labeling(db_path: str = DB_PATH, min_samples: int = 50,
                     last_days: Optional[int] = None,
                     export_formats: List[str] = ['json', 'conll', 'training']):
    """
    Полный пайплайн автоматической разметки

    Args:
        db_path: путь к БД
        min_samples: минимальное количество записей
        last_days: загрузить только за последние N дней
        export_formats: форматы экспорта (json, conll, training)
    """
    logger.info("🚀 Запуск автоматической разметки")

    # 1. Загрузка данных
    data = load_receipt_data(db_path, min_samples, last_days)
    if not data:
        logger.error("❌ Нет данных для разметки")
        return None

    # 2. Создание BIO-тегов
    bio_sequences = create_bio_sequences(data)
    if not bio_sequences:
        logger.error("❌ Не удалось создать BIO-теги")
        return None

    # 3. Статистика
    print_statistics(data, bio_sequences)

    # 4. Экспорт
    exported_files = []

    if 'json' in export_formats:
        path = export_bio_json(bio_sequences)
        exported_files.append(path)

    if 'conll' in export_formats:
        path = export_conll_format(bio_sequences)
        exported_files.append(path)

    if 'training' in export_formats:
        path = export_training_format(bio_sequences)
        exported_files.append(path)

    logger.info(f"✅ Автоматическая разметка завершена")
    logger.info(f"📁 Экспортировано файлов: {len(exported_files)}")

    return {
        'total_records': len(data),
        'labeled_sequences': len(bio_sequences),
        'exported_files': exported_files
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Автоматическая разметка данных для NER')
    parser.add_argument('--db', type=str, default=DB_PATH, help='Путь к БД')
    parser.add_argument('--min-samples', type=int, default=50, help='Минимальное количество записей')
    parser.add_argument('--last-days', type=int, help='Загрузить только за последние N дней')
    parser.add_argument('--format', type=str, nargs='+', default=['json', 'conll', 'training'],
                       help='Форматы экспорта: json, conll, training')

    args = parser.parse_args()

    result = run_auto_labeling(
        db_path=args.db,
        min_samples=args.min_samples,
        last_days=args.last_days,
        export_formats=args.format
    )

    if result:
        print(f"\n✅ Готово! Размечено {result['labeled_sequences']} последовательностей")
    else:
        print("\n❌ Ошибка автоматической разметки")
