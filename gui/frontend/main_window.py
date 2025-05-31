import sys
from pathlib import Path

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QPushButton, \
    QFileDialog, QFrame, QFormLayout, QHBoxLayout

from gui.frontend.image_window import ImageWindow


class MainWindow(QMainWindow):
    upload_image_signal = pyqtSignal(str)
    generic_signal = pyqtSignal()
    download_results_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.__create_widgets()
        self.__create_layouts()
        self.__setup_connections()
        self.setWindowTitle("Cell Detector")

        self.cv_result_path = ''
        self.ml_result_path = ''
        self.cnn_result_path = ''

    def __create_widgets(self):
        self.image_label = QLabel()
        self.image_label.setFixedSize(500, 375)
        self.image_label.setFrameShape(QFrame.Box)  # Рамка вокруг изображения
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Изображение")
        self.image_label.setStyleSheet("background-color: #f0f0f0; font-size: 16px;")

        self.cv_count_label = QLabel('-')
        self.ml_count_label = QLabel('-')
        self.cnn_count_label = QLabel('-')

        self.show_cv_result_button = QPushButton('Показать результат работы CV алгоритма')
        self.show_cv_result_button.setVisible(False)
        self.show_ml_result_button = QPushButton('Показать результат работы ML алгоритма')
        self.show_ml_result_button.setVisible(False)
        self.show_cnn_result_button = QPushButton('Показать результат работы CNN алгоритма')
        self.show_cnn_result_button.setVisible(False)

        self.upload_button = QPushButton('Загрузить изображение')
        self.generic_button = QPushButton('Сгенерировать изображение')
        self.download_experiments_button = QPushButton('Выгрузить результаты экспериментов')

    def __create_layouts(self):
        form_layout = QFormLayout()
        form_layout.addRow('CV  - ', self.cv_count_label)
        form_layout.addRow('ML  - ', self.ml_count_label)
        form_layout.addRow('CNN - ', self.cnn_count_label)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.show_cv_result_button)
        buttons_layout.addWidget(self.show_ml_result_button)
        buttons_layout.addWidget(self.show_cnn_result_button)

        bottom_layout = QHBoxLayout()
        bottom_layout.addLayout(form_layout)
        bottom_layout.addLayout(buttons_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(bottom_layout)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.generic_button)
        layout.addWidget(self.download_experiments_button)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def __setup_connections(self):
        self.upload_button.clicked.connect(self.open_file_dialog)
        self.show_cv_result_button.clicked.connect(self.show_cv_result_image)
        self.show_ml_result_button.clicked.connect(self.show_ml_result_image)
        self.show_cnn_result_button.clicked.connect(self.show_cnn_result_image)
        self.generic_button.clicked.connect(self.generic_signal)
        self.download_experiments_button.clicked.connect(self.download_results_signal)

    def open_file_dialog(self):
        file_dialog = QFileDialog()
        selected_file, _ = file_dialog.getOpenFileName(
            None,
            'Выбрать изображение',
            '.',
            "Image files (*.jpg *.jpeg *.png)")
        if not selected_file:
            return

        self.show_base_image(selected_file)
        self.upload_image_signal.emit(selected_file)

    def show_base_image(self, file_path: str):
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap.scaled(500, 375, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_results(self, cv_path, ml_path, cnn_path, cv_count, ml_count, cnn_count):
        self.cv_count_label.setText('-')
        self.ml_count_label.setText('-')
        self.cnn_count_label.setText('-')
        if cv_count >= 0:
            self.cv_count_label.setText(str(cv_count))
            if cv_path:
                self.show_cv_result_button.setVisible(True)
                self.cv_result_path = cv_path
        if ml_count >= 0:
            self.ml_count_label.setText(str(ml_count))
            if ml_path:
                self.show_ml_result_button.setVisible(True)
                self.ml_result_path = ml_path
        if cnn_count >= 0:
            self.cnn_count_label.setText(str(cnn_count))
            if cnn_path:
                self.show_cnn_result_button.setVisible(True)
                self.cnn_result_path = cnn_path

    def show_cv_result_image(self):
        self.result_image_window = ImageWindow(self.cv_result_path, 'CV')
        self.result_image_window.show()

    def show_ml_result_image(self):
        self.result_image_window = ImageWindow(self.ml_result_path, 'ML')
        self.result_image_window.show()

    def show_cnn_result_image(self):
        self.result_image_window = ImageWindow(self.cnn_result_path, 'CNN')
        self.result_image_window.show()





