import pandas as pd
from keras_preprocessing.image import ImageDataGenerator

from commons.config import STYLES_DATA_SET_PATH, IMAGES_FOLDER_PATH


class DataLoader:
    def __init__(self, target_column, batch_size=16, val_size=.1):
        print('Generating data for {0}...'.format(target_column))
        self._df = pd.read_csv(STYLES_DATA_SET_PATH, error_bad_lines=False)
        self._df.dropna()
        self._df['image'] = self._df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
        self._df = self._df[['image', target_column]]
        self._df.dropna()
        print('Targeting;\n')
        print(', '.join(self._df[target_column].unique().tolist()))

        self._val_size = val_size
        self._target_column = target_column
        self._batch_size = batch_size
        self._img_data_generator = ImageDataGenerator(
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=val_size
        )

    def steps_per_epoch_for_validation(self):
        return int(float(len(self._df) / self._batch_size) * self._val_size)

    def steps_per_epoch_for_training(self):
        return int(float(len(self._df) / self._batch_size) * (1. - self._val_size))

    def training_generator(self):
        return self._img_data_generator.flow_from_dataframe(
            dataframe=self._df,
            directory=IMAGES_FOLDER_PATH,
            x_col="image",
            y_col=self._target_column,
            target_size=(96, 96),
            batch_size=self._batch_size,
            subset="training"
        )

    def validation_generator(self):
        return self._img_data_generator.flow_from_dataframe(
            dataframe=self._df,
            directory=IMAGES_FOLDER_PATH,
            x_col="image",
            y_col=self._target_column,
            target_size=(96, 96),
            batch_size=self._batch_size,
            subset="validation"
        )
