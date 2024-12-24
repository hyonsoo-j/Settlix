from pathlib import Path
import torch

class Config:
    # Application settings
    APP_NAME = "Settlix"
    APP_VERSION = "1.1.1"
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # UI settings
    BASE_PATH = Path(__file__).resolve().parent.parent
    SPLASH_SCREEN_DURATION = 2000  # in milliseconds
    SPLASH_SCREEN_IMAGE_PATH = str(BASE_PATH / "resource" / "img" / "splash_img.png")
    UI_PATH = str(BASE_PATH / "resource" / "ui")

    # Graph settings
    GRAPH_WIDTH = 500

    # Data settings
    DATA_COLUMNS = ['settlement', 'fill_height']

    # Font settings
    FONT_PATH = "arial.ttf"
    FONT_SIZE = 12

    RANDOM_SEED = 1
    TRAIN_RATIO = 0.9
    MAX_NUM_EPOCHS = 1000
    PATIENCE = 10
