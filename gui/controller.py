from gui.backend.usecases import UseCases
from gui.frontend.main_window import MainWindow


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




