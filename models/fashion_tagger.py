from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


class FashionTagger:
    def __init__(self,
                 sub_category_count=36,
                 article_type_count=108,
                 gender_count=5,
                 season_count=4,
                 base_colour_count=45,
                 usage_count=6):
        self._sub_category_count = sub_category_count
        self._article_type_count = article_type_count
        self._gender_count = gender_count
        self._season_count = season_count
        self._base_colour_count = base_colour_count
        self._usage_count = usage_count

    def _sub_cat_branch(self, base_model):
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', name='sub_category_decider')(x)
        return Dense(self._sub_category_count, activation='softmax')(x)

    def _article_type_branch(self, base_model):
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', name='article_type_decider')(x)
        return Dense(self._article_type_count, activation='softmax')(x)

    def _gender_branch(self, base_model):
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', name='gender_decider')(x)
        return Dense(self._gender_count, activation='softmax')(x)

    def _season_branch(self, base_model):
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', name='season_decider')(x)
        return Dense(self._season_count, activation='softmax')(x)

    def _base_colour_branch(self, base_model):
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', name='base_colour_decider')(x)
        return Dense(self._base_colour_count, activation='softmax')(x)

    def _usage_branch(self, base_model):
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', name='usage_decider')(x)
        return Dense(self._usage_count, activation='softmax')(x)

    def generate_network(self):
        # create the base pre-trained model
        base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')

        predictions = list()
        predictions.append(self._sub_cat_branch(base_model))
        predictions.append(self._article_type_branch(base_model))
        predictions.append(self._gender_branch(base_model))
        predictions.append(self._base_colour_branch(base_model))
        predictions.append(self._usage_branch(base_model))
        predictions.append(self._season_branch(base_model))

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss=['categorical_crossentropy'] * 6, metrics=['accuracy'])
        print(model.summary())
        return model
