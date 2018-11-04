from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, UpSampling2D
from keras import Model
from keras.models import Model
from keras import backend as K
import mycoco
import numpy as np

input_img = Input(shape=(200, 200, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


mycoco.setmode('train')
zebraids, horseids = mycoco.query([['zebra'], ['horse']])
#len(zebraids), len(horseids), len(zebraids) + len(horseids)
imgs = mycoco.iter_images([zebraids, horseids], [0, 1], batch=10)

imgs = mycoco.iter_images([zebraids, horseids], [0, 1], batch=10, size=(200,200,1))
autoencoder.fit_generator(imgs, steps_per_epoch=4, epochs=4)

