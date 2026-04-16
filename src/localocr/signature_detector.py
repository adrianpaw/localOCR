"""
Модуль для детекции подписей на изображениях с использованием OpenCV.
Использует компьютерное зрение для обнаружения областей с подписями.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
from PIL import Image
import tempfile
import os

logger = logging.getLogger(__name__)


class SignatureDetector:
    """Детектор подписей на основе компьютерного зрения."""
    
    def __init__(self, min_signature_area: int = 500, max_signature_area: int = 50000):
        """
        Инициализация детектора подписей.
        
        Args:
            min_signature_area: Минимальная площадь области для рассмотрения как подпись (пиксели)
            max_signature_area: Максимальная площадь области для рассмотрения как подпись (пиксели)
        """
        self.min_signature_area = min_signature_area
        self.max_signature_area = max_signature_area
        
        # Параметры для эвристик
        self.aspect_ratio_range = (0.3, 3.0)  # диапазон соотношения сторон
        self.line_density_threshold = 0.15  # минимальная плотность линий
        self.solidity_threshold = 0.5  # минимальная сплошность контура
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения для улучшения детекции подписей.
        
        Args:
            image: Входное изображение в формате BGR или RGB
            
        Returns:
            Обработанное изображение в градациях серого
        """
        # Конвертируем в градации серого, если нужно
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Улучшение контраста с помощью CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Уменьшение шума с помощью медианного фильтра
        denoised = cv2.medianBlur(enhanced, 3)
        
        return denoised
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Обнаружение краев на изображении.
        
        Args:
            image: Изображение в градациях серого
            
        Returns:
            Бинарное изображение с краями
        """
        # Используем детектор краев Canny
        edges = cv2.Canny(image, 50, 150)
        
        # Морфологические операции для улучшения краев
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        return edges
    
    def calculate_line_density(self, region: np.ndarray, edges: np.ndarray) -> float:
        """
        Вычисление плотности линий в области.
        
        Args:
            region: Маска области
            edges: Изображение с краями
            
        Returns:
            Плотность линий (отношение пикселей краев к площади области)
        """
        # Применяем маску к изображению с краями
        masked_edges = cv2.bitwise_and(edges, edges, mask=region)
        
        # Считаем количество ненулевых пикселей (краев)
        edge_pixels = np.count_nonzero(masked_edges)
        area = np.count_nonzero(region)
        
        if area == 0:
            return 0.0
        
        return edge_pixels / area
    
    def analyze_contour(self, contour: np.ndarray, edges: np.ndarray) -> Dict[str, Any]:
        """
        Анализ контура для определения, является ли он подписью.
        
        Args:
            contour: Контур для анализа
            edges: Изображение с краями
            
        Returns:
            Словарь с характеристиками и решением
        """
        # Вычисляем характеристики контура
        area = cv2.contourArea(contour)
        
        # Пропускаем слишком маленькие или слишком большие области
        if area < self.min_signature_area or area > self.max_signature_area:
            return {"is_signature": False, "reason": "area_out_of_range", "area": area}
        
        # Получаем ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Проверяем соотношение сторон
        min_ar, max_ar = self.aspect_ratio_range
        if aspect_ratio < min_ar or aspect_ratio > max_ar:
            return {"is_signature": False, "reason": "aspect_ratio", "aspect_ratio": aspect_ratio}
        
        # Создаем маску для области контура
        mask = np.zeros(edges.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Вычисляем плотность линий
        line_density = self.calculate_line_density(mask, edges)
        if line_density < self.line_density_threshold:
            return {"is_signature": False, "reason": "low_line_density", "line_density": line_density}
        
        # Вычисляем сплошность (отношение площади контура к площади выпуклой оболочки)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        if solidity < self.solidity_threshold:
            return {"is_signature": False, "reason": "low_solidity", "solidity": solidity}
        
        # Все проверки пройдены - вероятно, это подпись
        return {
            "is_signature": True,
            "bbox": (x, y, w, h),
            "area": area,
            "aspect_ratio": aspect_ratio,
            "line_density": line_density,
            "solidity": solidity,
            "center": (x + w // 2, y + h // 2)
        }
    
    def detect_signatures(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Основной метод для детекции подписей на изображении.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Список словарей с информацией о найденных подписях
        """
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Не удалось загрузить изображение: {image_path}")
            return []
        
        # Предобработка
        processed = self.preprocess_image(image)
        
        # Обнаружение краев
        edges = self.detect_edges(processed)
        
        # Находим контуры
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logger.info(f"Найдено {len(contours)} контуров на изображении")
        
        # Анализируем каждый контур
        signatures = []
        for i, contour in enumerate(contours):
            analysis = self.analyze_contour(contour, edges)
            
            if analysis["is_signature"]:
                signatures.append({
                    "id": i,
                    "bbox": analysis["bbox"],
                    "area": analysis["area"],
                    "aspect_ratio": analysis["aspect_ratio"],
                    "line_density": analysis["line_density"],
                    "solidity": analysis["solidity"],
                    "center": analysis["center"]
                })
                logger.info(f"Контур {i}: обнаружена подпись (площадь: {analysis['area']:.0f})")
            else:
                logger.debug(f"Контур {i}: не подпись ({analysis.get('reason', 'unknown')})")
        
        logger.info(f"Обнаружено {len(signatures)} потенциальных подписей")
        return signatures
    
    def draw_detections(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Визуализация обнаруженных подписей на изображении.
        
        Args:
            image_path: Путь к исходному изображению
            output_path: Путь для сохранения результата (опционально)
            
        Returns:
            Изображение с отрисованными обнаружениями
        """
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Обнаруживаем подписи
        signatures = self.detect_signatures(image_path)
        
        # Рисуем ограничивающие прямоугольники
        for sig in signatures:
            x, y, w, h = sig["bbox"]
            # Рисуем зеленый прямоугольник для подписей
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Добавляем текст с ID
            cv2.putText(image, f"Signature {sig['id']}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Добавляем информацию о количестве найденных подписей
        cv2.putText(image, f"Found {len(signatures)} signatures", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Сохраняем результат, если указан путь
        if output_path:
            cv2.imwrite(output_path, image)
            logger.info(f"Результат сохранен в: {output_path}")
        
        return image
    
    def extract_signature_regions(self, image_path: str, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Извлечение областей с подписями из изображения.
        
        Args:
            image_path: Путь к исходному изображению
            output_dir: Директория для сохранения извлеченных областей (опционально)
            
        Returns:
            Список словарей с информацией об извлеченных областях
        """
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Обнаруживаем подписи
        signatures = self.detect_signatures(image_path)
        
        extracted_regions = []
        
        for i, sig in enumerate(signatures):
            x, y, w, h = sig["bbox"]
            
            # Извлекаем область с небольшим отступом
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            region = image[y1:y2, x1:x2]
            
            # Сохраняем область, если указана директория
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                region_path = os.path.join(output_dir, f"signature_{i}.png")
                cv2.imwrite(region_path, region)
                sig["region_path"] = region_path
            
            # Добавляем данные об извлеченной области
            region_data = {
                "id": i,
                "bbox": (x, y, w, h),
                "padded_bbox": (x1, y1, x2 - x1, y2 - y1),
                "region": region,
                "area": sig["area"]
            }
            
            if output_dir:
                region_data["region_path"] = os.path.join(output_dir, f"signature_{i}.png")
            
            extracted_regions.append(region_data)
        
        logger.info(f"Извлечено {len(extracted_regions)} областей с подписями")
        return extracted_regions


def test_detector():
    """Тестирование детектора на примере."""
    import matplotlib.pyplot as plt
    
    # Создаем тестовое изображение с простой "подписью"
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Рисуем простую подпись (волнистая линия)
    cv2.line(test_image, (50, 150), (350, 150), (0, 0, 0), 3)
    for i in range(10):
        x = 50 + i * 30
        y = 150 + 20 * np.sin(i * 0.5)
        cv2.circle(test_image, (int(x), int(y)), 2, (0, 0, 0), -1)
    
    # Сохраняем тестовое изображение
    test_path = "test_signature.png"
    cv2.imwrite(test_path, test_image)
    
    # Тестируем детектор
    detector = SignatureDetector()
    signatures = detector.detect_signatures(test_path)
    
    print(f"Найдено подписей: {len(signatures)}")
    for sig in signatures:
        print(f"  Подпись: bbox={sig['bbox']}, area={sig['area']:.0f}")
    
    # Визуализируем результат
    result = detector.draw_detections(test_path, "test_result.png")
    
    # Удаляем временные файлы
    os.remove(test_path)
    
    return len(signatures) > 0


if __name__ == "__main__":
    # Простой тест
    success = test_detector()
    print(f"Тест {'пройден' if success else 'не пройден'}")