import os
from random import choice

import cv2
import numpy as np

from config import base_dir, output_cells_dir, output_background_dir, output_filled_cells_dir, \
    output_filled_background_dir, original_dir, masked_dir, generated_orig_path, generated_mask_path


def generate():
    if not os.path.exists(masked_dir):
        pass

    filled_backs_filename = choice(os.listdir(output_filled_background_dir))
    transparent_cells_filename = choice(os.listdir(output_cells_dir))


    background = cv2.imread(os.path.join(output_filled_background_dir, filled_backs_filename))
    cells = cv2.imread(os.path.join(output_cells_dir, transparent_cells_filename), cv2.IMREAD_UNCHANGED)

    if background is None or cells is None:
        raise ValueError("Не удалось загрузить изображения")

    # 1. Случайное отражение
    flip_code = choice([-1, 0, 1, None])  # -1, 0, 1 или нет отражения
    if flip_code is not None:
        cells = cv2.flip(cells, flip_code)

    # 2. Наложение клеток на фон
    # Разделяем цветовые каналы и альфа-канал
    cells_bgr = cells[:, :, :3]
    alpha = cells[:, :, 3] / 255.0  # Нормализуем альфа-канал [0,1]

    # Создаем маску для наложения
    alpha_3ch = cv2.merge([alpha, alpha, alpha])

    # Наложение с учетом прозрачности
    result = (background * (1 - alpha_3ch) + cells_bgr * alpha_3ch).astype(np.uint8)

    # 3. Создаем бинарную маску
    _, binary_mask = cv2.threshold(alpha, 0.01, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)

    filename = os.path.splitext(filled_backs_filename)[0] + "---" + os.path.splitext(transparent_cells_filename)[0] + ".png"
    generated_orig_full_path = os.path.join(generated_orig_path, filename)
    generated_mask_full_path = os.path.join(generated_mask_path, filename)

    # Сохраняем результаты
    cv2.imwrite(generated_orig_full_path, result)
    cv2.imwrite(generated_mask_full_path, binary_mask)