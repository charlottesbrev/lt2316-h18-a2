from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, UpSampling2D, SpatialDropout2D
from keras import Model
from keras.models import Model
from keras import backend as K
import mycoco
import numpy as np

def test_func(a):
	return [a,a]

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
autoencoder.fit_generator(test_func(imgs), steps_per_epoch=4, epochs=4)

