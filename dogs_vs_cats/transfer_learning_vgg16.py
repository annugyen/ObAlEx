import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

#load data
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
m_train = np.load("m_train.npy")
m_test = np.load("m_test.npy")

#data augmentation to reduce overfitting
preprocess_fn = tf.keras.applications.vgg16.preprocess_input
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True,
                                    fill_mode = "nearest",
                                    preprocessing_function = preprocess_fn)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = preprocess_fn)

train_generator = train_datagen.flow(x_train, y_train)
test_generator = test_datagen.flow(x_test, y_test)

#load VGG16 without final dense layers for transfer learning
conv_base = tf.keras.applications.VGG16(weights = "imagenet", include_top = False,
                                        input_shape = (224, 224, 3))

#layer weigths of vgg16 should not be trained
conv_base.trainable = False

#final model with added dense layers
#softmax for multi class classification
flatten = tf.keras.layers.Flatten()(conv_base.output)
dense_1 = tf.keras.layers.Dense(256, activation = "relu", name = "dense_1")(flatten)
dense_2 = tf.keras.layers.Dense(2, name = "dense_2")(dense_1)
output = tf.keras.layers.Activation("softmax", name = "softmax")(dense_2)
model = tf.keras.models.Model(inputs = conv_base.inputs, outputs = output)

#optimizer and compile model
opt = tf.keras.optimizers.RMSprop(lr = 1e-5)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["acc"])

#train_steps * batch_size = number of images in train set
train_steps = len(x_train) // train_generator.batch_size
val_steps = len(x_test) // test_generator.batch_size

#train model for 10 epochs
history = model.fit_generator(train_generator, steps_per_epoch = train_steps, epochs = 10,
                    validation_data = test_generator,validation_steps = val_steps)
model.save("vgg16.h5")