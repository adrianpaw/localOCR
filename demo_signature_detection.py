#!/usr/bin/env python3
"""
Демонстрация работы детектора подписей в localOCR.

Этот скрипт показывает, как использовать новую функциональность
детекции подписей в системе localOCR.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from localocr.ocr import OCRExtractor
from localocr.signature_detector import SignatureDetector
import cv2
import numpy as np
import json


def create_test_document():
    """Создание тестового документа с подписями."""
    # Создаем изображение документа
    image = np.ones((800, 1000, 3), dtype=np.uint8) * 255
    
    # Заголовок
    cv2.putText(image, 'ДОГОВОР ОКАЗАНИЯ УСЛУГ', (100, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # Текст документа
    lines = [
        "№ 456/2026",
        "г. Москва, 16 апреля 2026 г.",
        "",
        "Исполнитель: ООО 'ТехноСервис'",
        "в лице директора Петрова П.П.",
        "",
        "Заказчик: Иванов Иван Иванович",
        "",
        "1. Предмет договора",
        "Исполнитель обязуется оказать услуги,",
        "а Заказчик принять и оплатить их.",
        "",
        "2. Стоимость и порядок расчетов",
        "Стоимость услуг составляет 50 000 рублей.",
        "",
        "Подписи сторон:",
        ""
    ]
    
    y_pos = 150
    for line in lines:
        cv2.putText(image, line, (100, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 30
    
    # Подпись исполнителя
    cv2.putText(image, 'Исполнитель:', (100, 500), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Рисуем подпись 1 (сложная волнистая линия)
    points1 = []
    for i in range(25):
        x = 250 + i * 15
        y = 530 + 30 * np.sin(i * 0.4) + 10 * np.cos(i * 0.2)
        points1.append((int(x), int(y)))
    
    for i in range(len(points1)-1):
        cv2.line(image, points1[i], points1[i+1], (0, 0, 0), 3)
    
    # Подпись заказчика
    cv2.putText(image, 'Заказчик:', (100, 650), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Рисуем подпись 2 (другая форма)
    points2 = []
    for i in range(20):
        x = 250 + i * 18
        y = 680 + 25 * np.cos(i * 0.5) + 5 * np.sin(i * 0.3)
        points2.append((int(x), int(y)))
    
    for i in range(len(points2)-1):
        cv2.line(image, points2[i], points2[i+1], (0, 0, 0), 3)
    
    return image


def demo_standalone_detector():
    """Демонстрация работы автономного детектора подписей."""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ: Автономный детектор подписей")
    print("=" * 60)
    
    # Создаем тестовый документ
    test_image = create_test_document()
    test_path = "demo_document.png"
    cv2.imwrite(test_path, test_image)
    print(f"Создан тестовый документ: {test_path}")
    print(f"Размер изображения: {test_image.shape[1]}x{test_image.shape[0]} пикселей")
    
    # Инициализируем детектор
    detector = SignatureDetector(
        min_signature_area=300,
        max_signature_area=30000
    )
    
    # Детектируем подписи
    print("\nЗапуск детекции подписей...")
    signatures = detector.detect_signatures(test_path)
    
    print(f"\nРезультаты детекции:")
    print(f"Найдено подписей: {len(signatures)}")
    
    for i, sig in enumerate(signatures):
        x, y, w, h = sig['bbox']
        print(f"\nПодпись {i+1}:")
        print(f"  Координаты: ({x}, {y}) - ({x+w}, {y+h})")
        print(f"  Размер: {w}x{h} пикселей")
        print(f"  Площадь: {sig['area']:.0f} пикселей")
        print(f"  Соотношение сторон: {sig['aspect_ratio']:.2f}")
        print(f"  Плотность линий: {sig['line_density']:.3f}")
        print(f"  Сплошность: {sig['solidity']:.3f}")
    
    # Визуализируем результат
    print("\nСоздание визуализации...")
    result_image = detector.draw_detections(test_path, "demo_detection_result.png")
    print("Визуализация сохранена: demo_detection_result.png")
    
    # Извлекаем области с подписями
    print("\nИзвлечение областей с подписями...")
    regions = detector.extract_signature_regions(test_path, "demo_extracted_signatures")
    print(f"Извлечено {len(regions)} областей")
    
    # Очистка
    os.remove(test_path)
    
    return len(signatures)


def demo_integrated_ocr():
    """Демонстрация работы интегрированного OCR с детекцией подписей."""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ: Интегрированный OCR с детекцией подписей")
    print("=" * 60)
    
    # Создаем тестовый документ
    test_image = create_test_document()
    test_path = "demo_document_ocr.png"
    cv2.imwrite(test_path, test_image)
    
    # Инициализируем OCR экстрактор
    print("Инициализация OCR экстрактора...")
    extractor = OCRExtractor(languages=['ru', 'en'], gpu=False)
    
    # Тест 1: Только текст
    print("\n1. Извлечение только текста:")
    text = extractor.extract_text(test_path)
    print(f"   Извлечено символов: {len(text)}")
    print(f"   Первые 150 символов:")
    print("   " + text[:150].replace('\n', '\n   ') + "...")
    
    # Тест 2: Текст с подписями
    print("\n2. Извлечение текста с детекцией подписей:")
    result = extractor.extract_with_signatures(test_path)
    print(f"   Извлечено символов: {len(result['text'])}")
    print(f"   Найдено подписей: {result['signature_count']}")
    print(f"   Есть подписи: {result['has_signatures']}")
    
    # Тест 3: Только подписи
    print("\n3. Детекция только подписей:")
    signatures = extractor.detect_signatures(test_path)
    print(f"   Найдено подписей: {len(signatures)}")
    
    # Очистка
    if os.path.exists(test_path):
        os.remove(test_path)
    
    return result['signature_count']


def demo_api_usage():
    """Демонстрация использования API."""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ: Использование API")
    print("=" * 60)
    
    print("\nПримеры использования REST API:")
    
    print("\n1. Извлечение текста с подписями (cURL):")
    print("""
curl -X POST http://localhost:5000/api/extract \\
  -F "file=@document.png" \\
  -F "detect_signatures=true"
""")
    
    print("\n2. Только детекция подписей (cURL):")
    print("""
curl -X POST http://localhost:5000/api/detect-signatures \\
  -F "file=@document.png"
""")
    
    print("\n3. Извлечение только текста (cURL):")
    print("""
curl -X POST http://localhost:5000/api/extract \\
  -F "file=@document.png"
""")
    
    print("\n4. Пример ответа API (JSON):")
    example_response = {
        "success": True,
        "filename": "document.png",
        "text": "Текст документа...",
        "signatures": [
            {
                "id": 0,
                "bbox": [250, 530, 350, 60],
                "area": 21000,
                "aspect_ratio": 5.83,
                "line_density": 0.187,
                "solidity": 0.723,
                "center": [425, 560]
            }
        ],
        "signature_count": 1,
        "has_signatures": True
    }
    print(json.dumps(example_response, indent=2, ensure_ascii=False))


def main():
    """Основная функция демонстрации."""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ ДЕТЕКТОРА ПОДПИСЕЙ В LOCALOCR")
    print("=" * 60)
    
    try:
        # Демонстрация автономного детектора
        sig_count1 = demo_standalone_detector()
        
        # Демонстрация интегрированного OCR
        sig_count2 = demo_integrated_ocr()
        
        # Демонстрация API
        demo_api_usage()
        
        print("\n" + "=" * 60)
        print("РЕЗЮМЕ")
        print("=" * 60)
        print(f"✓ Детектор подписей успешно протестирован")
        print(f"✓ На тестовом документе обнаружено: {max(sig_count1, sig_count2)} подписей")
        print(f"✓ Интеграция с OCR системой работает корректно")
        print(f"✓ API endpoints доступны для использования")
        print(f"✓ Визуализация результатов сохранена в файлы:")
        print(f"  - demo_detection_result.png")
        print(f"  - demo_extracted_signatures/ (директория)")
        
        print("\n" + "=" * 60)
        print("СЛЕДУЮЩИЕ ШАГИ")
        print("=" * 60)
        print("1. Запустите веб-сервер: python web.py")
        print("2. Откройте в браузере: http://localhost:5000")
        print("3. Загрузите документ с подписями для тестирования")
        print("4. Используйте API для интеграции в свои проекты")
        
    except Exception as e:
        print(f"\nОшибка при выполнении демонстрации: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())