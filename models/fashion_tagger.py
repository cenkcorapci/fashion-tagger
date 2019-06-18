import keras
from keras.applications.densenet import DenseNet201
from keras.layers import Dense, Dropout, GlobalMaxPooling2D
from keras.models import Model

from commons.config import IMAGE_SIZE
from commons.ai_utils import hn_multilabel_loss


class FashionTagger:
    def __init__(self, target_count):
        self._target = target_count

    def generate_network(self):
        # base pre-trained model
        base_model = DenseNet201(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights="imagenet", include_top=False)
        for layer in base_model.layers:
            layer.trainable = True
        x = base_model.output

        # added layers
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='selu', kernel_initializer='lecun_normal')(x)

        predictions = Dense(self._target, activation="sigmoid")(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        opt = keras.optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)

        model.compile(optimizer=opt, loss=hn_multilabel_loss, metrics=['accuracy', 'categorical_accuracy'])
        print(model.summary())
        return model
