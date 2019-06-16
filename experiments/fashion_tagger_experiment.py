# -*- coding: utf-8 -*-
from __future__ import print_function, division

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split

from commons.ai_utils import step_decay_schedule
from commons.config import *
from commons.config import TB_LOGS_PATH
from data.data_set import ClassificationDataSet


class FashionTaggerExperiment:
    def __init__(self, df,
                 model_name,
                 model,
                 val_split=0.1,
                 nb_epochs=3,
                 batch_size=256):
        self._nb_epochs = nb_epochs

        self._model_name = model_name
        self._model = model.generate_network()

        self._train_set, self._val_set = train_test_split(df, test_size=val_split)
        self._train_set = ClassificationDataSet(self._train_set,
                                                batch_size=batch_size,
                                                shuffle_on_end=True,
                                                do_augmentations=True)
        self._val_set = ClassificationDataSet(self._val_set,
                                              batch_size=batch_size,
                                              shuffle_on_end=False,
                                              do_augmentations=False)

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
        lr_scheduler = step_decay_schedule(initial_lr=.02, decay_factor=.75, step_size=2)
        self._callbacks = [tb_callback, es_callback, checkpoint_callback, lr_scheduler]

    def train_model(self):
        self._model.fit_generator(
            generator=self._train_set,
            callbacks=self._callbacks,
            validation_data=self._val_set,
            epochs=self._nb_epochs,
            class_weight=self._train_set.get_class_weights(),
            use_multiprocessing=True,
            workers=4,
            max_queue_size=32,
            verbose=1)

        logging.info("Training complete for {0}".format(self._model_name))
