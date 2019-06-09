import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from commons.config import STYLES_DATA_SET_PATH, IMAGES_FOLDER_PATH, RANDOM_STATE


class DataLoader:
    def __init__(self, batch_size=16, val_split=.1):
        df = pd.read_csv(STYLES_DATA_SET_PATH, error_bad_lines=False)
        df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
        self._df_train, self._df_val = train_test_split(df, test_size=val_split, random_state=RANDOM_STATE)

        self._batch_size = batch_size

    def training_generator(self):
        image_generator = ImageDataGenerator(
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True,
        )

        return image_generator.flow_from_dataframe(
            dataframe=self._df_train,
            directory=IMAGES_FOLDER_PATH,
            x_col="image",
            class_mode='other',
            y_col=list(["subCategory", "articleType", "gender", "baseColour", "usage", "season"]),
            target_size=(96, 96),
            batch_size=self._batch_size
        )

    def validation_generator(self):
        image_generator = ImageDataGenerator()

        return image_generator.flow_from_dataframe(
            dataframe=self._df_val,
            directory=IMAGES_FOLDER_PATH,
            x_col="image",
            y_col=list(["subCategory", "articleType", "gender", "baseColour", "usage", "season"]),
            class_mode='other',
            target_size=(96, 96),
            batch_size=self._batch_size
        )
