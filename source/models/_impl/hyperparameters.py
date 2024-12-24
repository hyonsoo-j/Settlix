class Hyperparameters:
    def __init__(self, model_properties):
        self.INPUT_SIZE = model_properties["INPUT_SIZE"]
        self.HIDDEN_SIZE = model_properties["HIDDEN_SIZE"]
        self.NUM_LAYERS = model_properties["NUM_LAYERS"]
        self.OUTPUT_SIZE = model_properties["OUTPUT_SIZE"]
        self.WINDOW_SIZE = model_properties["WINDOW_SIZE"]
        self.LEARNING_RATE = model_properties["LEARNING_RATE"]
        self.BEST_LOSS = model_properties["BEST_LOSS"]
        self.BEST_EPOCH = model_properties["BEST_EPOCH"]
        self.BATCH_SIZE = model_properties["BATCH_SIZE"]
