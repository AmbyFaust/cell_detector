# cell_detector
### Состав команды
1. Шарипов Тимур Наилевич
2. Баранов Владислав Васильевич

### Цель проекта
Определить количество клеток на изображении с помощью трёх методов:
1. CV
2. ML
3. CNN

### Первичный анализ
Исходные данные:
1. Датасет, состоящий из тренировочных и тестовых изображений и их масок для обучения моделей.


### Описание архитектуры проекта
Основное приложение выполнено с использованием `PyQt5`, выполнено разделение между `frontend` частью и `backend` частью,
которые связаны `PyqtSignal`'s.

```python
class Controller:
    def __init__(self):
        self.__create_widgets()
        self.__create_usecases()
        self.__setup_connections()
        
    def __create_widgets(self):
        self.main_window = MainWindow()

    def __create_usecases(self):
        self.usecases = UseCases()

    def __setup_connections(self):
        self.main_window.upload_image_signal.connect(self.usecases.uploaded_image_usecase)
        self.main_window.generic_signal.connect(self.usecases.generic_usecase)
        self.main_window.download_results_signal.connect(self.usecases.download_results)

        self.usecases.cells_detected_signal.connect(self.main_window.show_results)
        self.usecases.image_generated_signal.connect(self.main_window.show_base_image)

    def start(self):
        self.main_window.show()
```

Реализованные алгоритмы лежат в отдельной папке [core](./core), генератор изображений в папке [generator](./generator).
Исходные и сгенерированные данные лежат в папке [data](./data).

### Описание методов
1. Генератор.  ...
```python
def generate():
    background, filled_backs_filename = get_back(output_filled_background_dir)
    cells, transparent_cells_filename = get_cells(output_cells_dir)

    if background is None or cells is None:
        raise ValueError("Не удалось загрузить изображения")

    # Случайное отражение
    flip_code = choice([-1, 0, 1, None])  # -1, 0, 1 или нет отражения
    if flip_code is not None:
        cells = cv2.flip(cells, flip_code)

    # Наложение клеток на фон
    # Разделяем цветовые каналы и альфа-канал
    cells_bgr = cells[:, :, :3]
    alpha = cells[:, :, 3] / 255.0  # Нормализуем альфа-канал [0,1]

    # Создаем маску для наложения
    alpha_3ch = cv2.merge([alpha, alpha, alpha])

    # Наложение с учетом прозрачности
    result = (background * (1 - alpha_3ch) + cells_bgr * alpha_3ch).astype(np.uint8)

    # Создаем бинарную маску
    _, binary_mask = cv2.threshold(alpha, 0.01, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)

    filename = os.path.splitext(filled_backs_filename)[0] + "---" + os.path.splitext(transparent_cells_filename)[0] + ".png"
    generated_orig_full_path = os.path.join(generated_orig_path, filename)
    generated_mask_full_path = os.path.join(generated_mask_path, filename)

    # Сохраняем результаты
    cv2.imwrite(generated_orig_full_path, result)
    cv2.imwrite(generated_mask_full_path, binary_mask)

    return generated_orig_full_path
```
2. CV. Первично - получение бинарной маски с удалением шумов, затем определение контуров, 
для каждого из которых производится подсчёт количества кругов, которые он содержит, с помощью метода `cv2.HoughCircles`
```python
def cv_detector(image_path) -> (str, int):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=0)
    _, threshold = cv2.threshold(opening, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []

    for cnt in contours:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, 1)

        circles_tmp = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 30, param1=150, param2=15, minRadius=30,
                                       maxRadius=65)

        if circles_tmp is not None:
            circles_tmp = np.round(circles_tmp[0, :]).astype("int")
            for c in circles_tmp:
                if mask[c[1] - 1][c[0] - 1] == 0:
                    circles.append(c)

    circles = [circles]
    counts = len(circles[0])
```
3. ML. ...
4. CNN. Модель CNN состоит из Encoder и Decoder, каждый из которых имеет несколько слоёв.
```python
class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.down1 = self._block(3, 32, 3)
        self.down2 = self._block(32, 64, 3)
        self.down3 = self._block(64, 128, 3)

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = self._block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = self._block(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def _block(self, in_channels, out_channels, kernel=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
```
CNN модель обучена по изображению создавать его маску, на которую в дальнейшем натравливается простой cv алгоритм "поиска областей связности".


### Результаты, полученные в ходе оптимизации моделей
...

### Результаты исполнения
1. Генератор. Генератор способен выдавать изображения с разным типом клеток, разными размерами клеток, с поворотами и наложением друг на друга,
также способен выдавать фон с естественным шумом, свойственным клеточной плазме.
2. CV. Алгоритм довольно точно определяет одиночные клетки, но имеет трудности с участками наложения клеток друг на друга,
также шум в местах малого удаления клеток друг от друга может привести к слипанию клеток при получении маски, что тоже ухудшает точность результатов алгоритма.
3. ML. ...
4. CNN. Изображение, проходя через модель, выдаёт крайне неплохую маску, позволяющую простому алгоритму CV
находить и определять центры клеток.





