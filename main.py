import os
import sys
import traceback
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt  

from source.gui import MainWindow, SplashScreen
from source.config import Config

def main():
    try:
        app = QApplication(sys.argv)

        splash_pix = QPixmap(Config.SPLASH_SCREEN_IMAGE_PATH)
        splash_pix = splash_pix.scaled(600, 400, Qt.KeepAspectRatio)

        splash = SplashScreen(splash_pix, Config.SPLASH_SCREEN_DURATION)
        splash.show()

        main_window = MainWindow()
        splash.show_splash(main_window)

        sys.exit(app.exec_())
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error: {e}")
        print("Traceback (most recent call last):")
        print(tb)

if __name__ == "__main__":
    main()
