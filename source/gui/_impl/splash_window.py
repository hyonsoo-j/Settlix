from PyQt5.QtWidgets import QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

class SplashScreen(QSplashScreen):
    def __init__(self, pixmap, duration):
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.duration = duration

    def show_splash(self, main_window):
        self.show()
        QTimer.singleShot(self.duration, self.finish_splash(main_window))

    def finish_splash(self, main_window):
        def callback():
            self.finish(main_window)
            main_window.show()
        return callback

    def mousePressEvent(self, event):
        pass
