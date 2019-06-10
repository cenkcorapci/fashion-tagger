# -*- coding: utf-8 -*-
from __future__ import print_function, division

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from commons.config import *
from commons.config import TB_LOGS_PATH
from data.data_loader import DataLoader
from models.fashion_tagger import FashionTagger


class FashionTaggerExperiment:
    def __init__(self, target, val_split=0.1, nb_epochs=3, batch_size=256):
        self._in_inference_mode = False
        self._nb_epochs = nb_epochs
        self._model_name = 'densenet_121_fashion_' + target

        self._loader = DataLoader(target, batch_size, val_split)

        self._train_data_set = self._loader.training_generator()
        self._val_data_set = self._loader.validation_generator()
        self._model = FashionTagger(target)
        self._model = self._model.generate_network()
        tb_log_dir = TB_LOGS_PATH + self._model_name + '/'
        pathlib.Path(tb_log_dir).mkdir(parents=True, exist_ok=True)

        # Set training/validation callbacks
        tb_callback = TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_graph=True,
                                  write_images=False,
                                  embeddings_freq=0,
                                  embeddings_metadata=None)
        es_callback = EarlyStopping(patience=5, monitor='val_loss')
        checkpoint_callback = ModelCheckpoint(DL_MODELS_PATH + self._model_name + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=False,
                                              save_weights_only=False)

        self._callbacks = [tb_callback, es_callback, checkpoint_callback]

    def train_model(self):
        self._model.fit_generator(
            generator=self._train_data_set,
            callbacks=self._callbacks,
            steps_per_epoch=self._loader.steps_per_epoch_for_training(),
            validation_data=self._val_data_set,
            validation_steps=self._loader.steps_per_epoch_for_validation(),
            epochs=self._nb_epochs,
            verbose=1)

        logging.info("Training complete for {0}".format(self._model_name))
