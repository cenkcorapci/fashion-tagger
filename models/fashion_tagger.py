import keras
from keras.applications.densenet import DenseNet201
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Dropout, GlobalMaxPooling2D, Flatten
from keras.models import Model

from commons.config import IMAGE_SIZE


class FashionTaggerModels:
    dense201 = 'dense201'
    mobilenetv2 = 'mobilenetv2'


class FashionTagger:
    def __init__(self, target_count, model=FashionTaggerModels.mobilenetv2):
        self._target = target_count
        self._model = model

    def generate_network(self):
        base_model = None
        # base pre-trained model
        if self._model == FashionTaggerModels.mobilenetv2:
            base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights="imagenet", include_top=False)
            for layer in base_model.layers:
                layer.trainable = True
        elif self._model == FashionTaggerModels.dense201:
            base_model = DenseNet201(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights="imagenet", include_top=False)
            for layer in base_model.layers:
                layer.trainable = False
        x = base_model.output

        # added layers
        x = GlobalMaxPooling2D()(x) if self._model == FashionTaggerModels.mobilenetv2 else Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(4096 if self._model == FashionTaggerModels.dense201 else 1024,
                  activation='selu', kernel_initializer='lecun_normal')(x)
        if self._model == FashionTaggerModels.dense201:
            x = Dropout(0.5)(x)
            x = Dense(2048, activation='selu', kernel_initializer='lecun_normal')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='selu', kernel_initializer='lecun_normal')(x)

        predictions = Dense(self._target, activation="softmax")(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        opt = keras.optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model
