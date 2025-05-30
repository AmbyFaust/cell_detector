from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel


class ImageWindow(QWidget):
    def __init__(self, image_path: str, type_title: str):
        super().__init__()
        self.setWindowTitle(f"Изображение {type_title}")

        layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        if image_path:
            self.load_image(image_path)
        else:
            self.image_label.setText("Изображение не загружено")
            self.image_label.setStyleSheet("background-color: #f0f0f0; font-size: 16px;")

        layout.addWidget(self.image_label)

    def load_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(500, 375, Qt.KeepAspectRatio, Qt.SmoothTransformation))
