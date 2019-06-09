# -*- coding: utf-8 -*-
from __future__ import print_function, division

from keras.callbacks import TensorBoard

from commons.config import *
from commons.config import TB_LOGS_PATH
from data.data_loader import DataLoader
from models.fashion_tagger import FashionTagger
from commons.model_storage import ModelStorage


class FashionTaggerExperiment:
    def __init__(self, val_split=0.1, nb_epochs=3, learning_rate=0.01, batch_size=32):
        self._in_inference_mode = False
        self._nb_epochs = nb_epochs
        self._lr = learning_rate
        self._model_name = 'mobile_net_fashion_multi_loss'

        logging.info("Getting data set...")
        loader = DataLoader(batch_size, val_split)

        self._train_data_set = loader.training_generator()
        self._val_data_set = loader.validation_generator()
        self._model = FashionTagger()
        self._model = self._model.generate_network()

        # Set training/validation callbacks
        tb_callback = TensorBoard(log_dir=TB_LOGS_PATH + self._model_name + '/', histogram_freq=0, write_graph=True,
                                  write_images=False,
                                  embeddings_freq=0,
                                  embeddings_metadata=None)
        self._callbacks = [tb_callback]
        self.model_storage = ModelStorage(self._model_name)

    def train_model(self):
        self._model.fit_generator(
            generator=self._train_data_set,
            validation_data=self._val_data_set,
            epochs=self._nb_epochs,
            verbose=1)

        logging.info("Training complete")
        self.model_storage.persist(self._model)
        logging.info("Model persisted")


if __name__ == "__main__":
    experiment = FashionTaggerExperiment(nb_epochs=2, val_split=.1)
