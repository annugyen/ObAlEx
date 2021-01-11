import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import transform

img_files = sorted(os.listdir("cats_vs_dogs/images"))
mask_files = sorted(os.listdir("cats_vs_dogs/masks"))
print("{} image files found".format(len(img_files)))
print("{} mask files found".format(len(mask_files)))

#load images with size 224x224
images = []
for i in range(len(img_files)):
  img = tf.keras.preprocessing.image.load_img(os.path.join("cats_vs_dogs/images", img_files[i]), target_size = (224, 224))
  img = tf.keras.preprocessing.image.img_to_array(img, dtype = "uint8")
  images.append(img)

#1.500 images per class
cat_images = images[:1500]
dog_images = images[12500:14000]

#load masks
cat_masks = np.load("drive/My Drive/Colab/Data/cat_masks.npy")
dog_masks = np.load("drive/My Drive/Colab/Data/dog_masks.npy")

#resize masks to match images size
def resize_masks(masks):
  resized_masks = []
  for i in masks:
    i = transform.resize(i, (224, 224))
    i[i > 0] = 255
    i = i.astype("uint8")
    resized_masks.append(i)
  return resized_masks

cat_masks = resize_masks(cat_masks)
dog_masks = resize_masks(dog_masks)


#detect images with no masks
no_cat_masks = []
for i in range(len(cat_images)):
  if cat_masks[i].any() == 0:
    no_cat_masks.append(i)
print("no masks avaliable for {} cat images".format(len(no_cat_masks)))

no_dog_masks = []
for i in range(len(dog_images)):
  if dog_masks[i].any() == 0:
    no_dog_masks.append(i)
print("no masks avaliable for {} dog images".format(len(no_dog_masks)))

#remove images with no masks from dataset
for i in sorted(no_cat_masks, reverse = True):
    del cat_images[i]
    del cat_masks[i]

for i in sorted(no_dog_masks, reverse = True):
    del dog_images[i]
    del dog_masks[i]

print("{} cat images with masks".format(len(cat_images)))
print("{} dog images with masks".format(len(dog_images)))

#concatenate image arrays
x = cat_images + dog_images
x = np.array(x, dtype = "uint8")

#labels: 1-dimensional array with 0 for cat, 1 for dog
y_cat = np.zeros(len(cat_images), dtype = "uint8")
y_dog = np.ones(len(dog_images), dtype = "uint8")
y = np.concatenate((y_cat, y_dog))

#masks for train and test data
m = cat_masks + dog_masks
m = np.array(m)

#split dataset in train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
#shuffle masks in same order
m_train, m_test, _, _ = train_test_split(m, y, test_size = 0.33, random_state = 42)

#cat: [1, 0] - dog: [0, 1]
#two output neurons for gradCAM
y_train = tf.keras.utils.to_categorical(y_train, num_classes = 2, dtype = "uint8")
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 2, dtype = "uint8")

print("{} train images".format(len(x_train)))
print("{} test images".format(len(x_test)))

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)
np.save("m_train.npy", m_train)
np.save("m_test.npy", m_test)