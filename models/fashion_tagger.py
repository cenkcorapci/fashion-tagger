from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adamax
from commons.config import CATEGORY_COUNTS_DICT


class FashionTagger:
    def __init__(self,
                 target):
        self._target = target

    def _decider(self, x):
        x = Dense(1024, activation='relu', name='usage_decider')(x)
        return Dense(self._usage_count, activation='softmax')(x)

    def generate_network(self):
        # create the base pre-trained model
        base_model = VGG16(input_shape=(96, 96, 3), weights="imagenet", include_top=False)

        x = base_model.output

        # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
        for layer in base_model.layers[:5]:
            layer.trainable = False

        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="selu", kernel_initializer='lecun_normal')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="selu", kernel_initializer='lecun_normal')(x)

        predictions = Dense(CATEGORY_COUNTS_DICT[self._target], activation="softmax")(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        opt = Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model
