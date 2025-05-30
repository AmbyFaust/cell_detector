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
        self.usecases.cells_detected_signal.connect(self.main_window.show_results)

    def start(self):
        self.main_window.show()




