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

import random
x_train_masked = x_train.copy()
for i, img in enumerate(tqdm(x_train_masked)):
  m = m_train[i]
  for y in range(224):
    for x in range(224):
      if m[x, y] == 0:
        img[x, y] = [random.randrange(255), random.randrange(255), random.randrange(255)]
np.save("drive/My Drive/x_train_masked.npy", x_train_masked)

#comment out if using masked images for training
#x_train = np.load("x_train_masked.npy")

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

from tensorflow.keras import backend as K
from scipy.ndimage.interpolation import zoom

def gradcam(input_model, image, class_index, layer_name = "block5_conv3"):
    height = image.shape[0]
    image = np.expand_dims(image, axis = 0)
    y_c = input_model.output[0, class_index]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    grads = ((grads + 1e-10) / (K.sqrt(K.mean(K.square(grads))) + 1e-10))
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = zoom(cam, height / cam.shape[0])
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

def score(explanation, mask):
  if np.sum(explanation) != 0:
    return np.sum(explanation[mask == 255]) / np.sum(explanation)
  else:
    return 0

def avg_score(x, m, y):
  scores = []
  indices = []
  #get indices of first 50 correctly classified images
  i = 0
  while len(indices) < 50 and i < len(x):
    prediction = model.predict(np.expand_dims(preprocess_fn(x[i]), axis = 0))
    if np.argmax(prediction) == np.argmax(y[i]):
      indices.append(i)
    i += 1
  for i in tqdm(indices):
    img = x[i]
    mask = m[i]
    class_index = np.argmax(y[i])
    img_preprocessed = preprocess_fn(img)
    explanation = gradcam(model, img_preprocessed, class_index)
    scores.append(score(explanation, mask))
  return np.mean(scores)

opt = tf.keras.optimizers.RMSprop(lr = 1e-3)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["acc"])

gradcam_train = [avg_score(x_train, m_train, y_train)]
gradcam_test = [avg_score(x_test, m_test, y_test)]
eval_train = model.evaluate_generator(train_generator)
eval_test = model.evaluate_generator(test_generator)
acc = [eval_train[1]]
val_acc = [eval_test[1]]
loss = [eval_train[0]]
val_loss = [eval_test[0]]

for i in range(10):
  history = model.fit_generator(train_generator, steps_per_epoch = train_steps, epochs = 1,
                                validation_data = test_generator,validation_steps = val_steps)
  gradcam_train.append(avg_score(x_train, m_train, y_train))
  gradcam_test.append(avg_score(x_test, m_test, y_test))
  acc.append(history.history["acc"][0])
  val_acc.append(history.history["val_acc"][0])
  loss.append(history.history["loss"][0])
  val_loss.append(history.history["val_loss"][0])


plt.figure(figsize = (14, 4.5))
label_fs = 16
tick_fs = 14
plt.subplot(121)
plt.plot(acc, linewidth = 3, linestyle = ":", color = "tab:cyan", label = "acc")
plt.plot(val_acc, linewidth = 3, color = "tab:cyan", label = "test acc")
plt.xlabel("Epoch", fontsize = label_fs)
plt.ylabel("Accuracy", fontsize = label_fs)
plt.xticks(range(0, 11), fontsize = tick_fs)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize = tick_fs)
plt.ylim(0, 1)
plt.legend()

plt.subplot(122)
plt.plot(gradcam_train, color = "tab:cyan", linestyle = ":", linewidth = 3, label = "gradcam train")
plt.plot(gradcam_test, color = "tab:cyan", linestyle = "-", linewidth = 3, label = "gradcam test")
plt.xlabel("Epoch", fontsize = label_fs)
plt.ylabel(r"$AvgScore_E$", fontsize = label_fs)
plt.ylim(0.50, 0.75)
plt.xticks(range(0, 11), fontsize = tick_fs)
plt.yticks(fontsize = tick_fs)
plt.legend()
plt.show()