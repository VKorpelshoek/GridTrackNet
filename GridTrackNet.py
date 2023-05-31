from keras.models import *
from keras.layers import *
from keras.activations import *

def conv_layer(inputs, filters, kernel_size):
    x = Conv2D(filters, kernel_size, padding='same', data_format='channels_first')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def GridTrackNet(imgs_per_instance, input_height, input_width ):
		
	imgs_input = Input(shape=(imgs_per_instance*3,input_height,input_width))

	x = conv_layer(imgs_input, 64, (3,3))

	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	x = conv_layer(x, 128, (3,3))

	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	x = conv_layer(x, 256, (3,3))

	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	x = conv_layer(x, 512, (3,3))

	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	output = Conv2D(imgs_per_instance*3, (3,3), activation='sigmoid', data_format='channels_first', padding='same')(x)

	model = Model(imgs_input, output)

	return model