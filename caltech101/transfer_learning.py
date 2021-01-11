import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.notebook import tqdm
import os

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
m_train = np.load("m_train.npy")
m_test = np.load("m_test.npy")

class_labels = ['accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus',
                'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body',
                'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin',
                'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'Faces', 'Faces_easy', 'ferry', 'flamingo',
                'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter',
                'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'Leopards', 'llama', 'lobster', 'lotus',
                'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'Motorbikes', 'nautilus', 'octopus', 'okapi', 'pagoda',
                'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors',
                'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry',
                'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair',
                'wrench', 'yin_yang']

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

train_steps = len(x_train) // train_generator.batch_size
val_steps = len(x_test) // test_generator.batch_size

conv_base = tf.keras.applications.VGG16(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))
conv_base.trainable = False

x = tf.keras.layers.Flatten()(conv_base.output)
x = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_1")(x)
x = tf.keras.layers.Dense(101, name = "classification")(x)
x = tf.keras.layers.Softmax(name = "softmax")(x)
model = tf.keras.models.Model(inputs = conv_base.inputs, outputs = x)

#train model for 10 epochs
history = model.fit_generator(train_generator, steps_per_epoch = train_steps, epochs = 10,
                    validation_data = test_generator,validation_steps = val_steps)
model.save("vgg16.h5")