import os

import cv2
from tqdm import tqdm


def split_transparent(original_dir, masked_dir, output_cells_dir, output_background_dir, overwrite = False):
    """
    Разделяет клетки и фон, оставляя прозрачными места разделений
    """
    original_files = os.listdir(original_dir)
    not_found = []
    skipped = 0
    affected = 0
    not_found_cnt = 0
    volume = len(original_files)

    for filename in tqdm(original_files):
        original_path = os.path.join(original_dir, filename)
        mask_path = os.path.join(masked_dir, filename)

        if not os.path.exists(mask_path):
            not_found.append(filename)
            not_found_cnt += 1
            continue

        output_cells_path = os.path.join(output_cells_dir, os.path.splitext(filename)[0] + ".png")
        output_background_path = os.path.join(output_background_dir, os.path.splitext(filename)[0] + ".png")

        if os.path.exists(output_cells_path) or os.path.exists(output_background_path):
            if not overwrite:
                skipped += 1
                continue

        # Читаем изображения
        original = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # маска в grayscale

        # Если оригинал в формате без альфа-канала, конвертируем в BGRA (чтобы добавить прозрачность)
        if original.shape[2] == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)

        # Бинаризуем маску (чтобы гарантировать, что она чёрно-белая)
        _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Инвертируем маску для фона
        inverted_mask = cv2.bitwise_not(binary_mask)

        # Создаём изображение с клетками (фон прозрачный)
        cells = original.copy()
        cells[:, :, 3] = binary_mask  # альфа-канал = маска (клетки видны, фон прозрачный)

        # Создаём изображение с фоном (клетки прозрачные)
        background = original.copy()
        background[:, :, 3] = inverted_mask  # альфа-канал = инвертированная маска (фон виден, клетки прозрачные)

        # Сохраняем результаты (в PNG, чтобы сохранить прозрачность)
        cv2.imwrite(output_cells_path, cells)
        cv2.imwrite(output_background_path, background)
        affected += 1

    print("Обработка завершена!")
    print(f"Всего изображений: {volume}, из них:\n\tНовых создано: {affected},\n\tПропущено как существующий: {skipped},\n\tНе найдено маски: {not_found_cnt}")
