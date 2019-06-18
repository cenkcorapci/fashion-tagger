import logging
import random as pyrandom
from random import randint

import imgaug.augmenters as iaa
import keras
import numpy as np
from PIL import Image
from keras.preprocessing import image
from sklearn.utils import shuffle

from commons.config import IMAGES_FOLDER_PATH
from commons.config import IMAGE_SIZE
from commons.data_utils import generate_get_data_set
from commons.image_utils import scale_image


class TaggingDataSet(keras.utils.Sequence):
    def __init__(self,
                 df,
                 targets,
                 batch_size=128,
                 shuffle_on_end=True,
                 do_augmentations=True):
        self._batch_size = batch_size
        self._shuffle = shuffle_on_end
        self._target_list = targets
        self._data_set = generate_get_data_set(df)
        self._do_augmentations = do_augmentations
        logging.info('{0} targets; {1}'.format(len(targets), ', '.join(self._target_list)))

        self._aug = iaa.Sequential([
            iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.Flipud(0.5),  # vertically flip 50% of the images
            iaa.GaussianBlur(sigma=(0., 2.))  # blur images with a sigma of 0 to 3.0
        ])

        self.on_epoch_end()

    def __len__(self):
        return int(len(self._data_set) / self._batch_size)

    def __getitem__(self, index, random=False):
        try:
            i = pyrandom.randint(0, int(self.__len__() / self._batch_size)) if random else index
            samples = self._data_set[i * self._batch_size:(i + 1) * self._batch_size]
            X, y = self.__data_generation(samples)
            return X, y
        except Exception as exp:
            logging.error("Can't fetch batch #{0}, fetching a random batch.".format(index), exp)
            if not random:
                return self.__getitem__(index, random=True)  # Get a random batch if an error occurs
            else:
                raise exp

    def on_epoch_end(self):
        if self._shuffle:
            logging.info("Shuffling data set")
            shuffle(self._data_set)

    def __data_generation(self, samples):
        X = np.empty((self._batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
        y = np.zeros((self._batch_size, len(self._target_list)))

        for i, sample in enumerate(samples):
            img_path, labels = sample[0], sample[1]
            X[i] = self._load_image(IMAGES_FOLDER_PATH + img_path)
            for label in labels:
                y[i][self._target_list.index(label)] = 1.
        return X, y

    def _load_image(self, img_path):
        f = open(img_path, 'rb')
        f = Image.open(f)
        f = scale_image(f, [IMAGE_SIZE, IMAGE_SIZE])
        f = np.asarray(f)
        img_data = image.img_to_array(f)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = img_data.astype('float32') / 255.
        img_data = np.clip(img_data, 0., 1.)
        img_data = img_data[0]
        if self._do_augmentations and randint(0, 1) == 1:
            img_data = self._aug.augment_image(img_data)
        return img_data
