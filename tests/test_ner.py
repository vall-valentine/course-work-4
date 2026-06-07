# -*- coding: utf-8 -*-
"""
Тесты для NER модуля.

Важно: файл сохранён в UTF-8, чтобы русские строки корректно отображались и
корректно сравнивались в ассертах.
"""

import os
import sys

import pytest


# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestServiceLineFilter:
    @pytest.mark.parametrize(
        "line,expected",
        [
            # Служебные строки
            ("НДС 20%", True),
            ("ндс 10%", True),
            ("ИТОГО: 1250.00", True),
            ("ИТОГ 1250.00", True),
            ("ВСЕГО: 1250.00", True),
            ("СУММА: 1250.00", True),
            ("Оплата картой", True),
            ("КАССИР: Иванова А.А.", True),
            # Товарные строки (не должны считаться служебными)
            ("Молоко 3.2% Простоквашино 1л", False),
            ("Хлеб бородинский 400г", False),
            ("Сыр Российский 200г", False),
            ("Яйца С0 10шт", False),
            ("Пакет майка", False),
            # Граничные случаи
            ("", False),
            ("   ", False),
            ("123", False),
            ("НДС", True),
        ],
    )
    def test_service_line_detection(self, line, expected):
        from model_service import is_service_line

        assert is_service_line(line) == expected

    def test_trim_service_tail_in_mixed_line(self):
        from model_service import split_by_service_keyword

        line = "Молоко Простоквашино 1л НДС 20% 89.90"
        assert split_by_service_keyword(line) == "Молоко Простоквашино 1л"

    def test_trim_service_tail_for_service_only_line(self):
        from model_service import split_by_service_keyword

        assert split_by_service_keyword("НДС 20% 89.90") is None

    def test_service_line_with_whitespace(self):
        from model_service import is_service_line

        assert is_service_line("  НДС 20%  ") is True
        assert is_service_line("  Молоко  ") is False


class TestTextCleaning:
    @pytest.mark.parametrize(
        "input_text,expected_words",
        [
            ("Молоко 3.2% 1л", ["молоко"]),
            ("Хлеб бородинский 400г", ["хлеб", "бородинский"]),
            ("Пакет #1", ["пакет"]),
            ("Товар @магазин", ["товар", "магазин"]),
            ("Молоко 1 шт", ["молоко"]),
            ("Сыр 200г 150р", ["сыр"]),
            ("Кофе  Lavazza 250г", ["кофе", "lavazza"]),
        ],
    )
    def test_clean_text_removes_noise(self, input_text, expected_words):
        from model_service import clean_text

        result = clean_text(input_text)
        for word in expected_words:
            assert word in result

        assert not any(char.isdigit() for char in result)


class TestAutoLabeling:
    def test_find_product_position_exact(self):
        from auto_labeling import find_product_position

        raw_line = "Молоко 3.2% Простоквашино 1л    89.90"
        product = "молоко"

        result = find_product_position(raw_line, product)
        assert result is not None

        start, end = result
        assert raw_line[start:end].lower().strip() == product

    def test_find_product_position_not_found(self):
        from auto_labeling import find_product_position

        raw_line = "Пакет майка большой"
        product = "молоко"
        assert find_product_position(raw_line, product) is None


class TestNERPrediction:
    def test_ner_predict_empty(self):
        from model_service import ner_predict_single

        assert ner_predict_single("") == []

    def test_ner_predict_short_text(self):
        from model_service import ner_predict_single

        assert ner_predict_single("аб") == []


class TestIntegration:
    def test_full_pipeline(self):
        from model_service import is_fully_service_line, ner_predict_single, split_by_service_keyword

        lines = [
            "Молоко 3.2% Простоквашино 1л    89.90",
            "Хлеб бородинский 400г           45.00",
            "ИТОГО: 134.90",
            "НДС 20%",
        ]

        products = []
        for line in lines:
            trimmed = split_by_service_keyword(line)
            if not trimmed or is_fully_service_line(trimmed):
                continue
            products.extend(ner_predict_single(trimmed))

        assert len(products) > 0
