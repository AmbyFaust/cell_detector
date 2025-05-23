import os

import cv2
import numpy as np
from tqdm import tqdm

from generator.utils import create_tiled_background


def get_filled_cells(original_dir, masked_dir, output_filled_cells_dir, overwrite = False):
    original_files = os.listdir(original_dir)
    not_found = []
    skipped = 0
    affected = 0
    not_found_cnt = 0
    volume = len(original_files)

    for filename in tqdm(original_files):
        original_path = os.path.join(original_dir, filename)
        mask_path = os.path.join(masked_dir, filename)

        # Проверяем, существует ли маска
        if not os.path.exists(mask_path):
            not_found.append(filename)
            not_found_cnt += 1
            continue

        output_cells_path = os.path.join(output_filled_cells_dir, os.path.splitext(filename)[0] + ".png")

        if os.path.exists(output_cells_path):
            if not overwrite:
                skipped += 1
                continue

        # Читаем изображения
        original = cv2.imread(original_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # маска в grayscale

        # Бинаризуем маску
        _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        inverted_mask = cv2.bitwise_not(binary_mask)

        cells = original.copy()
        white_background = np.ones_like(cells) * 255
        cells = cv2.bitwise_and(cells, cells, mask=binary_mask) + cv2.bitwise_and(white_background, white_background,
                                                                                  mask=inverted_mask)
        cv2.imwrite(output_cells_path, cells)
        affected += 1

    print("Обработка завершена!")
    print(
        f"Всего изображений: {volume}, из них:\n\tНовых создано: {affected},\n\tПропущено как существующий: {skipped},\n\tНе найдено маски: {not_found_cnt}")



def get_filled_back(original_dir, masked_dir, output_filled_background_dir, overwrite = False):
    original_files = os.listdir(original_dir)
    not_found = []
    skipped = 0
    affected = 0
    not_found_cnt = 0
    error = 0
    volume = len(original_files)

    for filename in tqdm(original_files):
        original_path = os.path.join(original_dir, filename)
        mask_path = os.path.join(masked_dir, filename)

        # Проверяем, существует ли маска
        if not os.path.exists(mask_path):
            not_found.append(filename)
            not_found_cnt += 1
            continue

        output_background_path = os.path.join(output_filled_background_dir, os.path.splitext(filename)[0] + ".png")

        if os.path.exists(output_background_path):
            if not overwrite:
                skipped += 1
                continue

        # Читаем изображения
        original = cv2.imread(original_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # маска в grayscale

        # Бинаризуем маску
        _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        inverted_mask = cv2.bitwise_not(binary_mask)

        background = original.copy()
        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

        background[:, :, 3] = inverted_mask  # альфа-канал = инвертированная маска (фон виден, клетки прозрачные)

        try:
            background = create_tiled_background(background)
        except RuntimeError:
            print(f"error while processing file {filename}")
            error += 1
            continue

        cv2.imwrite(output_background_path, background)
        affected += 1

    print("Обработка завершена!")
    print(
        f"Всего изображений: {volume}, из них:\n\tНовых создано: {affected},\n\tПропущено как существующий: {skipped},\n\tНе найдено маски: {not_found_cnt},\n\tОшибка: {error}")