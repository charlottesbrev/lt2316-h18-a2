from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, UpSampling2D, SpatialDropout2D
from keras import Model
from keras.models import Model
from keras import backend as K
import mycoco
import numpy as np

def autoencoder_generator(iterator, batch_size):
    """
    Turns iterator of tuple(image, category) into generator of tuple(batch(image), batch(image))
    """
    while True:
        batch = []
        for b in range(batch_size):
            sample = next(iterator)
            batch.append(sample[0][0])
        result = array(batch)
        yield result, result

input_img = Input(shape=(200, 200, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((5, 5), padding='same')(x)
x = SpatialDropout2D(0, data_format='channels_last')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = SpatialDropout2D(0, data_format='channels_last')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

# (10, 10, 8) encoded
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = SpatialDropout2D(0, data_format='channels_last')(encoded)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = SpatialDropout2D(0, data_format='channels_last')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = SpatialDropout2D(0, data_format='channels_last')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((5, 5))(x)
x = SpatialDropout2D(0, data_format='channels_last')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_absolute_error')


mycoco.setmode('train')
zebraids, horseids = mycoco.query([['zebra'], ['horse']])
imgs = mycoco.iter_images([zebraids, horseids], [0, 1], batch=10, size=(200,200, 3))
r = autoencoder_generator(imgs, 4)
autoencoder.fit_generator(r, steps_per_epoch=4, epochs=4)

