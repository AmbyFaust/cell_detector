import os
from random import choice, random, randint
import uuid

import cv2
import numpy as np

from config import base_dir, output_cells_dir, output_background_dir, output_filled_cells_dir, \
    output_filled_background_dir, original_dir, masked_dir, generated_orig_path, generated_mask_path, min_circles, \
    max_circles, blue_range, purple_range, color_variation, min_radius, max_radius


def get_back(output_filled_background_directory):
    filled_backs_filename = choice(os.listdir(output_filled_background_directory))
    background = cv2.imread(os.path.join(output_filled_background_directory, filled_backs_filename))
    if random() < 0.5:
        return background, filled_backs_filename
    else:
        mean_color = cv2.mean(background)[:3]
        mean_background = np.ones_like(background) * mean_color
        return mean_background.astype(np.uint8), "color-" + uuid.uuid4().__str__() + ".png"

def get_cells(output_cells_directory):
    transparent_cells_filename = choice(os.listdir(output_cells_directory))
    cells = cv2.imread(os.path.join(output_cells_directory, transparent_cells_filename), cv2.IMREAD_UNCHANGED)
    if random() < 0.5:
        return cells, transparent_cells_filename
    else:
        height, width = cells.shape[:2]
        # Создаем прозрачный холст
        transparent_canvas = np.zeros((height, width, 4), dtype=np.uint8)

        num_circles = randint(min_circles, max_circles)

        base_blue = randint(*blue_range)
        base_red = randint(*purple_range)
        base_green = randint(0, min(base_blue, base_red) // 2)  # Мало зеленого для фиолетового
        base_color = (base_blue, base_green, base_red)  # BGR + Alpha

        for _ in range(num_circles):
            color_offset = np.array([
                randint(-color_variation, color_variation),
                randint(-color_variation, color_variation),
                randint(-color_variation, color_variation)
            ])

            circle_color = np.clip(base_color + color_offset, 0, 255)
            circle_color = np.append(circle_color, 255)  # Добавляем альфа-канал (непрозрачный)

            # Генерируем случайные параметры круга
            center_x = randint(0, width)
            center_y = randint(0, height)
            radius = randint(min_radius, max_radius)
            # Рисуем круг с заливкой
            cv2.circle(transparent_canvas, (center_x, center_y), radius, circle_color.tolist(), -1)

        return transparent_canvas, "color-" + uuid.uuid4().__str__() + ".png"

def generate():
    background, filled_backs_filename = get_back(output_filled_background_dir)
    cells, transparent_cells_filename = get_cells(output_cells_dir)

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