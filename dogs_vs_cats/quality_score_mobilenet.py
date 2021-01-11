import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tqdm.notebook import tqdm

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
m_train = np.load("m_train.npy")
m_test = np.load("m_test.npy")

preprocess_fn = tf.keras.applications.mobilenet.preprocess_input
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

#load model which was trained for 10 epochs
model = tf.keras.models.load_model("mobilenet.h5")
occlusion_model = tf.keras.models.Model([model.inputs], [model.get_layer("dense_2").output])

from tensorflow.keras import backend as K
from scipy.ndimage.interpolation import zoom

def gradcam(input_model, image, class_index, layer_name = "conv_pw_13_relu"):
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

def gradcam_plus(input_model, image, class_index, layer_name = "conv_pw_13_relu"):
    height = image.shape[0]
    image = np.expand_dims(image, axis = 0)
    y_c = input_model.output[0, class_index]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    grads = ((grads + 1e-10) / (K.sqrt(K.mean(K.square(grads))) + 1e-10))

    first = K.exp(y_c)*grads
    second = K.exp(y_c)*grads*grads
    third = K.exp(y_c)*grads*grads*grads

    gradient_function = K.function([input_model.input], [y_c,first,second,third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([image])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)
    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)
    alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))
    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
    cam = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    cam = zoom(cam, height / cam.shape[0])
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)  
    return cam
  
def occlusion(model, img, class_index):
  patch = 32
  stride = 16
  width = img.shape[0]
  cmap_width = int((width - patch) / stride + 1)
  cmap = np.zeros((cmap_width, cmap_width))

  for y in range(cmap_width):
    batch = []
    for x in range(cmap_width):
      occluded_img = img.copy()
      occluded_img[y * stride:y * stride + patch, x * stride:x * stride + patch] =  0
      batch.append(occluded_img)
    
    batch = np.array(batch)
    predictions = model.predict(batch)[:, class_index]
    for x in range(cmap_width):
      cmap[y, x] = predictions[x]
    
  cmap = zoom(cmap, width / cmap_width)
  cmap -= np.min(cmap)
  cmap /= (np.max(cmap) - np.min(cmap))
  cmap = 1 - cmap
  return cmap

import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
explainer = lime_image.LimeImageExplainer()

def lime(model, img, class_index):
  np.random.seed(42)
  explanation = explainer.explain_instance(img, mobilenet_predict)
  temp, mask = explanation.get_image_and_mask(class_index, positive_only = True, hide_rest = False, num_features = 5)
  return np.array(mask * 255, dtype = "uint8")

def mobilenet_predict(x):
  return model.predict(preprocess_fn(x))

def score(explanation, mask):
    if np.sum(explanation) != 0:
      return np.sum(explanation[mask == 255]) / np.sum(explanation)
    else:
      return 0

def avg_score(x, m, y, method):
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
    if method == "gradcam":
      explanation = gradcam(model, img_preprocessed, class_index)
    if method == "gradcam_plus":
      explanation = gradcam_plus(model, img_preprocessed, class_index)
    if method == "occlusion":
      explanation = occlusion(occlusion_model, img_preprocessed, class_index)
    if method == "lime":
      print("{}/50".format(i + 1))
      explanation = lime(model, img, class_index)
    s = score(explanation, mask)
    scores.append(s)
  return np.mean(scores)

#unfreeze layers depending on which strategy is trained
#dense
model.trainable = True
for layer in model.layers:
  layer.trainable = False
  if "dense" in layer.name:
    layer.trainable = True

#all
model.trainable = True

#block_4_5
model.trainable = True
for layer in model.layers:
  layer.trainable = False
for i in range(44, 87):
   model.layers[i].trainable = True

#block_1_2_3
model.trainable = True
for layer in model.layers:
  layer.trainable = False
for i in range(44):
  model.layers[i].trainable = True

opt = tf.keras.optimizers.RMSprop(lr = 1e-5)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["acc"])

gradcam_train = [avg_score(x_train, m_train, y_train, "gradcam")]
gradcam_test = [avg_score(x_test, m_test, y_test, "gradcam")]
gradcam_plus_train = [avg_score(x_train, m_train, y_train, "gradcam_plus")]
gradcam_plus_test = [avg_score(x_test, m_test, y_test, "gradcam_plus")]
occlusion_train = [avg_score(x_train, m_train, y_train, "occlusion")]
occlusion_test = [avg_score(x_test, m_test, y_test, "occlusion")]
lime_train = [avg_score(x_train, m_train, y_train, "lime")]
lime_test = [avg_score(x_test, m_test, y_test, "lime")]
eval_train = model.evaluate_generator(train_generator)
eval_test = model.evaluate_generator(test_generator)
acc = [eval_train[1]]
val_acc = [eval_test[1]]
loss = [eval_train[0]]
val_loss = [eval_test[0]]

for i in range(10):
  history = model.fit_generator(train_generator, steps_per_epoch = train_steps, epochs = 1,
                                  validation_data = test_generator,validation_steps = val_steps)
  gradcam_train.append(avg_score(x_train, m_train, y_train, "gradcam"))
  gradcam_test.append(avg_score(x_test, m_test, y_test, "gradcam"))
  gradcam_plus_train.append(avg_score(x_train, m_train, y_train, "gradcam_plus"))
  gradcam_plus_test.append(avg_score(x_test, m_test, y_test, "gradcam_plus"))
  occlusion_train.append(avg_score(x_train, m_train, y_train, "occlusion"))
  occlusion_test.append(avg_score(x_test, m_test, y_test, "occlusion"))
  lime_train.append(avg_score(x_train, m_train, y_train, "lime"))
  lime_test.append(avg_score(x_test, m_test, y_test, "lime"))
  acc.append(history.history["acc"][0])
  val_acc.append(history.history["val_acc"][0])
  loss.append(history.history["loss"][0])
  val_loss.append(history.history["val_loss"][0])


plt.figure(figsize = (10, 3.5))
label_fs = 14
title_fs = 16
tick_fs = 12
plt.subplot(121)
plt.plot(acc, linewidth = 2, linestyle = ":", color = "r", label = "acc")
plt.plot(val_acc, linewidth = 1, color = "r", label = "test acc", marker = "o")
plt.plot(loss, color = "b", linewidth = 2, linestyle = ":", label = "loss")
plt.plot(val_loss, color = "b", linewidth = 1, label = "test loss", marker = "^")
plt.xlabel("Epoch", fontsize = label_fs)
plt.ylabel("Accuracy / Loss", fontsize = label_fs)
plt.xticks(range(0, 11), fontsize = tick_fs)
plt.yticks(np.arange(0, 1.2, 0.2), fontsize = tick_fs)
plt.axhline(1, c = "black", ls = "--")

plt.subplot(122)
plt.plot(gradcam_train, color = "g", linestyle = ":", linewidth = 2, label = "gradcam train")
plt.plot(gradcam_test, color = "g", linestyle = "-", linewidth = 1, label = "gradcam test", marker = "o")
plt.plot(gradcam_plus_train, color = "c", linestyle = ":", linewidth = 2, label = "gradcam++ train")
plt.plot(gradcam_plus_test, color = "c", linestyle = "-", linewidth = 1, label = "gradcam++ test", marker = "^")
plt.plot(occlusion_train, color = "black", linestyle = ":", linewidth = 2, label = "occlusion train")
plt.plot(occlusion_test, color = "black", linestyle = "-", linewidth = 1, label = "occlusion test", marker = "o")
plt.plot(lime_train, color = "m", linestyle = ":", linewidth = 2, label = "lime train")
plt.plot(lime_test, color = "m", linestyle = "-", linewidth = 1, label = "lime test", marker = "v")
plt.xlabel("Epoch", fontsize = label_fs)
plt.ylabel("AvgScore", fontsize = label_fs)
plt.ylim(0.4, 0.725)
plt.xticks(range(0, 11), fontsize = tick_fs)
plt.yticks(np.arange(0.4, 0.8, 0.1), fontsize = tick_fs)
plt.show()