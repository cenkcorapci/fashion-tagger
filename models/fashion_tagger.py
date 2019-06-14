import keras
from keras.applications.xception import Xception
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.models import Model

from commons.config import CATEGORY_COUNTS_DICT


class FashionTagger:
    def __init__(self, target):
        self._target = target

    def generate_network(self):
        # base pre-trained model
        base_model = Xception(input_shape=(96, 96, 3), weights="imagenet", include_top=False)

        x = base_model.output
        for layer in base_model.layers:
            layer.trainable = False

        # added layers
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(CATEGORY_COUNTS_DICT[self._target], activation="softmax")(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        opt = keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model
