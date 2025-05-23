import cv2
import numpy as np
import random


def create_tiled_background(texture, num_patches=10, patch_size=100, blur_strength=2, color_threshold=17):
    if texture is None:
        raise ValueError("Не удалось загрузить текстуру")

    if texture.shape[2] != 4:
        raise ValueError("Текстура должна содержать альфа-канал")

    height, width = texture.shape[:2]

    alpha = texture[:, :, 3]
    opaque_mask = alpha > 200

    result = np.ones((height, width, 3), dtype=np.uint8) * 255

    patches = []
    attempts = 0
    max_attempts = 1000

    while len(patches) < num_patches and attempts < max_attempts:
        attempts += 1
        x = random.randint(0, width - patch_size)
        y = random.randint(0, height - patch_size)

        patch_mask = opaque_mask[y:y + patch_size, x:x + patch_size]
        if np.all(patch_mask):
            patch = texture[y:y + patch_size, x:x + patch_size, :3]
            # Если цвет в пределах допустимого отклонения
            if np.std(patch) < color_threshold:
                patches.append(patch)

    if not patches:
        raise RuntimeError("Не удалось найти непрозрачные области")

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = random.choice(patches)

            # Случайные трансформации
            if random.random() > 0.5:
                patch = cv2.flip(patch, 1)
            if random.random() > 0.5:
                patch = cv2.flip(patch, 0)
            if random.random() > 0.7:
                patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)

            # Вставка с учетом границ изображения
            end_i = min(i + patch_size, height)
            end_j = min(j + patch_size, width)
            patch_height = end_i - i
            patch_width = end_j - j

            result[i:end_i, j:end_j] = patch[:patch_height, :patch_width]

    grid_size = int(patch_size // 3 * blur_strength)  # Увеличиваем зону размытия

    # Создаем маску стыков
    blur_mask = np.zeros((height, width), dtype=np.uint8)

    # Вертикальные стыки (более толстые линии)
    for j in range(0, width, patch_size):
        cv2.line(blur_mask, (j, 0), (j, height), 255, grid_size)

    # Горизонтальные стыки (более толстые линии)
    for i in range(0, height, patch_size):
        cv2.line(blur_mask, (0, i), (width, i), 255, grid_size)

    # Сильное размытие маски
    blur_mask = cv2.GaussianBlur(blur_mask, (0, 0), sigmaX=grid_size)
    blur_mask = blur_mask.astype(np.float32) / 255.0

    # Двухэтапное размытие результата
    blurred_mild = cv2.GaussianBlur(result, (grid_size + 1, grid_size + 1), 0)
    blurred_strong = cv2.GaussianBlur(result, (grid_size * 2 + 1, grid_size * 2 + 1), 0)

    # Комбинируем с разной силой размытия
    result = (result * (1 - blur_mask[..., np.newaxis]) +
              blurred_mild * (blur_mask[..., np.newaxis] * 0.7) +
              blurred_strong * (blur_mask[..., np.newaxis] * 0.3))
    result = result.astype(np.uint8)

    # 4. Дополнительное микширование
    noise = np.random.normal(0, 8, result.shape).astype(np.uint8)
    result = cv2.addWeighted(result, 0.92, noise, 0.08, 0)

    # Дополнительное размытие по всей площади
    result = cv2.bilateralFilter(result, 5, 25, 25)

    return result