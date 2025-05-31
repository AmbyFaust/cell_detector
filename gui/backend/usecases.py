from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal

from core.cnn.cnn import cnn_detector
from core.cv import cv_detector
from core.db.db import Database
from core.ml.ml import ml_detector
from generator.main import generate_images


class UseCases(QObject):
    cells_detected_signal = pyqtSignal(str, str, str, int, int, int)
    image_generated_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.db = Database()

    def uploaded_image_usecase(self, file_path: str):
        cv_result_path, cv_count = self.cells_detect_cv(file_path)
        ml_result_path, ml_count = self.cells_detect_ml(file_path)
        cnn_result_path, cnn_count = self.cells_detect_cnn(file_path)
        self.db.add_experiment(file_path, 0, cv_count, ml_count, cnn_count)
        self.cells_detected_signal.emit(cv_result_path, ml_result_path, cnn_result_path, cv_count, ml_count, cnn_count)

    def cells_detect_cv(self, file_path: str) -> (str, int):
        return cv_detector(file_path)

    def cells_detect_ml(self, file_path: str) -> (str, int):
        return ml_detector(file_path)

    def cells_detect_cnn(self, file_path: str) -> (str, int):
        return cnn_detector(file_path)

    def generic_usecase(self):
        file_path = generate_images()
        self.image_generated_signal.emit(file_path)
        cv_result_path, cv_count = self.cells_detect_cv(file_path)
        ml_result_path, ml_count = self.cells_detect_ml(file_path)
        cnn_result_path, cnn_count = self.cells_detect_cnn(file_path)
        self.db.add_experiment(file_path, 1, cv_count, ml_count, cnn_count)
        self.cells_detected_signal.emit(cv_result_path, ml_result_path, cnn_result_path, cv_count, ml_count, cnn_count)

    def download_results(self):
        self.db.export_to_csv()


