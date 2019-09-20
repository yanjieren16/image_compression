from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from time import time
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

import sys


import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#x_train = x_train.astype('float32') / 127.5 - 1.
#x_test = x_test.astype('float32') / 127.5 - 1.


#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


decoder = load_model('/Users/yanjieren/PycharmProjects/image_compression/DC_gen.h5')
decoder.summary()
decoder.trainable = False
encoding_dim = 100  # 100 floats -> compression of factor 24.5, assuming the input is 784 floats


"""
////
input_img = Input(shape=(28, 28, 1))
# build encoder

x = Conv2D(64, kernel_size=3, activation='LeakyRelu', padding='same')(input_img)
x = Conv2D(128, kernel_size=3, activation='relu', strides=(2,2), padding='same')(x)
x = Conv2D(256, kernel_size=3, strides=(2,2), padding='same')(x)
x = Flatten()(x)
encoded = Dense(encoding_dim, activation='tanh')(x)
encoder = Model(input_img, encoded)

encoder.compile(optimizer='Adam', loss='binary_crossentropy')
z = encoder(input_img)
decoded = decoder(z)
////
"""


def build_encoder(img_shape, encoding_dim):
    model = Sequential()

    model.add(Conv2D(64, kernel_size = 3, padding = 'same', input_shape = (28,28,1)))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2D(128, kernel_size = 3,strides = (2,2), padding = 'same'))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2D(256, kernel_size= 3, strides=(2,2),padding = 'same'))
    model.add(Flatten())
    model.add(Dense(encoding_dim))
    model.add(Activation('tanh'))

    model.summary()

    img = Input(shape = img_shape)
    encoded = model(img)
    return Model(img, encoded)

img_shape = (28,28,1)
encoder = build_encoder(img_shape, encoding_dim)
optimizer = Adam(0.0002, 0.5)
encoder.compile(optimizer=optimizer, loss='binary_crossentropy')
input_img = Input(shape=(28, 28, 1))
z = encoder(input_img)
decoded = decoder(z)

# build autoencoder
AE = Model(input_img, decoded)
AE.compile(optimizer=optimizer, loss='binary_crossentropy')

# TensorBoard
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

AE.fit(x_train, x_train,
                epochs=20,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tensorboard])

# encode and decode some digits
# note that we take them from the *test* set

#encoded_imgs = encoder.predict(x_test)

# use Matplotlib
encoder.save('encoder.h5')

decoded_imgs = AE.predict(x_test)
#encoder = load_model('encoder.h5')
encoded_latent = encoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



