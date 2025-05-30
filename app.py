import sys

from PyQt5.QtWidgets import QApplication

from gui.controller import Controller

if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = Controller()
    controller.start()
    sys.exit(app.exec_())