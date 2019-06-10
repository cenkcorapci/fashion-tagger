from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model

from commons.config import CATEGORY_COUNTS_DICT


class FashionTagger:
    def __init__(self, target):
        self._target = target

    def generate_network(self):
        # create the base pre-trained model
        base_model = VGG19(input_shape=(96, 96, 3), weights="imagenet", include_top=False)

        x = base_model.output

        for layer in base_model.layers:
            layer.trainable = False

        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="selu", kernel_initializer='lecun_normal')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="selu", kernel_initializer='lecun_normal')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="selu", kernel_initializer='lecun_normal')(x)
        predictions = Dense(CATEGORY_COUNTS_DICT[self._target], activation="softmax")(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer='nadam', loss='categorical_crossentropy',
                      metrics=['accuracy', 'mse'])
        print(model.summary())
        return model
