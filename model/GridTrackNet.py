# from keras.models import *
# from keras.layers import *
# from keras.activations import *

# def GridTrackNet(imgs_per_instance, input_height, input_width ):

# 	imgs_input = Input(shape=(imgs_per_instance*3,input_height,input_width))

# 	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(imgs_input)
# 	x = ( Activation('relu'))(x)
# 	x = ( BatchNormalization())(x)

# 	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
# 	x = ( Activation('relu'))(x)
# 	x = ( BatchNormalization())(x)

# 	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

# 	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
# 	x = ( Activation('relu'))(x)
# 	x = ( BatchNormalization())(x)

# 	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
# 	x = ( Activation('relu'))(x)
# 	x = ( BatchNormalization())(x)

# 	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

# 	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
# 	x = ( Activation('relu'))(x)
# 	x = ( BatchNormalization())(x)

# 	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
# 	x = ( Activation('relu'))(x)
# 	x = ( BatchNormalization())(x)

# 	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

# 	x = ( Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
# 	x = ( Activation('relu'))(x)
# 	x = ( BatchNormalization())(x)

# 	x = ( Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
# 	x = ( Activation('relu'))(x)
# 	x = ( BatchNormalization())(x)

# 	x = ( Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
# 	x = ( Activation('relu'))(x)
# 	x = ( BatchNormalization())(x)

# 	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

# 	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
# 	x = ( Activation('relu'))(x)
# 	x = ( BatchNormalization())(x)

# 	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
# 	x = ( Activation('relu'))(x)
# 	x = ( BatchNormalization())(x)

# 	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
# 	x = ( Activation('relu'))(x)
# 	x = ( BatchNormalization())(x)

# 	x = ( Conv2D(imgs_per_instance*3, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
# 	output = ( Activation('sigmoid'))(x)

# 	model = Model( imgs_input , output)

# 	return model

from keras.models import *
from keras.layers import *
from keras.activations import *

def GridTrackNet(imgs_per_instance, input_height, input_width ):

	imgs_input = Input(shape=(imgs_per_instance*3,input_height,input_width))
	#x = tf.reshape(imgs_input,shape=(-1, 8*3, input_height, input_width))

	#Layer1
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(imgs_input)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	# #Layer2
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer3
	x = MaxPooling2D((4, 4), data_format='channels_first' )(x)

	#Layer7
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer8
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)


	#Layer10
	x = MaxPooling2D((4, 4), data_format='channels_first' )(x)

	#Layer11
	x = ( Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer12
	x = ( Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer12
	x = ( Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#output layer
	x = ( Conv2D(imgs_per_instance*3, (1, 1), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	output = ( Activation('sigmoid'))(x)

	#output = tf.reshape(x,shape=(-1, imgs_per_instance, input_height//16, input_width//16, 3))

	model = Model( imgs_input , output)

	#model.summary()

	return model







