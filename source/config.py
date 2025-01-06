from pathlib import Path
import torch
import sys

class Config:
    APP_NAME = "Settlix"
    APP_VERSION = "v.1.5"
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_base_path():
        if hasattr(sys, '_MEIPASS'):
            return Path(sys._MEIPASS)
        return Path(__file__).resolve().parent.parent

    BASE_PATH = get_base_path.__func__()

    SPLASH_SCREEN_DURATION = 2000  
    SPLASH_SCREEN_IMAGE_PATH = str(BASE_PATH / "resource" / "img" / "splash_img.png")
    UI_PATH = str(BASE_PATH / "resource" / "ui")

    GRAPH_WIDTH = 500

    DATA_COLUMNS = ['settlement', 'fill_height']

    FONT_PATH = str(BASE_PATH / "resource" / "fonts" / "arial.ttf") 
    FONT_SIZE = 12

    RANDOM_SEED = 1
    TRAIN_RATIO = 0.9
    MAX_NUM_EPOCHS = 1000
    PATIENCE = 10
