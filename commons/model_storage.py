import os

from commons.config import *


class ModelStorage:
    def __init__(self, model_name):
        self.model_name = model_name
        self.weights_file_path = DL_MODELS_PATH + self.model_name + '.h5'

    def persist(self, model):
        # serialize weights to HDF5
        model.save_weights(self.weights_file_path)
        print("Saved model to disk")

    def load(self, model):
        if not os.path.exists(self.weights_file_path):
            logging.error(
                "Can not find a pre-trained {0} weights on s3!".format(self.model_name))
            return None
        else:
            model.load_weights(self.weights_file_path)
            return model
