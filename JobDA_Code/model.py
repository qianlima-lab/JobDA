# Model: https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
import keras
import numpy as np
np.random.seed(813306)


class ResNet(object):
    def __init__(self, input_shape, nb_classes):
        self.input_shape = input_shape
        self.nb_classes = nb_classes


    def build_model(self):
        n_feature_maps = 64
        
        print ('build conv_x')
        x = keras.layers.Input(shape=(self.input_shape))
        conv_x = keras.layers.BatchNormalization()(x)
        conv_x = keras.layers.Conv2D(n_feature_maps, 8, padding='same')(conv_x)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        print ('build conv_y')
        conv_y = keras.layers.Conv2D(n_feature_maps, 5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        print ('build conv_z')
        conv_z = keras.layers.Conv2D(n_feature_maps, 3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        is_expand_channels = not (self.input_shape[-1] == n_feature_maps)
        if is_expand_channels:
            shortcut_y = keras.layers.Conv2D(n_feature_maps, 1, padding='same')(x)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
        else:
            shortcut_y = keras.layers.BatchNormalization()(x)
        print ('Merging skip connection')
        y = keras.layers.Add()([shortcut_y, conv_z])
        y = keras.layers.Activation('relu')(y)

        print ('build conv_x')
        x1 = y
        conv_x = keras.layers.Conv2D(n_feature_maps*2, 8, padding='same')(x1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        print ('build conv_y')
        conv_y = keras.layers.Conv2D(n_feature_maps*2, 5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        print ('build conv_z')
        conv_z = keras.layers.Conv2D(n_feature_maps*2, 3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        is_expand_channels = not (self.input_shape[-1] == n_feature_maps*2)
        if is_expand_channels:
            shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, padding='same')(x1)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
        else:
            shortcut_y = keras.layers.BatchNormalization()(x1)
        print ('Merging skip connection')
        y = keras.layers.Add()([shortcut_y, conv_z])
        y = keras.layers.Activation('relu')(y)
    
        print ('build conv_x')
        x1 = y
        conv_x = keras.layers.Conv2D(n_feature_maps*2, 8, padding='same')(x1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        print ('build conv_y')
        conv_y = keras.layers.Conv2D(n_feature_maps*2, 5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        print ('build conv_z')
        conv_z = keras.layers.Conv2D(n_feature_maps*2, 3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        is_expand_channels = not (self.input_shape[-1] == n_feature_maps*2)
        if is_expand_channels:
            shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, padding='same')(x1)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
        else:
            shortcut_y = keras.layers.BatchNormalization()(x1)
        print ('Merging skip connection')
        y = keras.layers.Add()([shortcut_y, conv_z])
        y = keras.layers.Activation('relu')(y)

        full = keras.layers.GlobalAveragePooling2D()(y)
        out = keras.layers.Dense(self.nb_classes, activation='softmax')(full)

        model = keras.models.Model(inputs=x, outputs=out)
        optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model
