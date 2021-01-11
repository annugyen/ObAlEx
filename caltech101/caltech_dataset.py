import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from PIL import Image
from tqdm import tqdm
import random

PATH = "caltech101"
#images
images_path = sorted(os.listdir(os.path.join(PATH, "101_ObjectCategories")), key = str.lower)
#annotations contain outline points of objects
annotations_path = sorted(os.listdir(os.path.join(PATH, "Annotations")), key = str.lower)

x = []
y = []
m = []

for j, object_class in enumerate(images_path):
    image_files = sorted(os.listdir(os.path.join(PATH, "101_ObjectCategories", object_class)), key = str.lower)
    mask_files =  sorted(os.listdir(os.path.join(PATH, "Annotations",  object_class)), key = str.lower)
    label = np.zeros(len(images_path), dtype = "uint8")
    label[j] = 1

    x_temp = []
    y_temp = []
    m_temp = []
    
    for i, img_file in enumerate(tqdm(image_files)):
        img = Image.open(os.path.join(PATH, "101_ObjectCategories", object_class, img_file)).convert("RGB")
        width, height = img.size
        img = img.resize((224, 224))
        img = np.array(img, dtype = "uint8")

        #convert object outline points to object mask
        annotation = scipy.io.loadmat(os.path.join(PATH, "Annotations", object_class, mask_files[i]))
        b = annotation["box_coord"][0]
        o = annotation["obj_contour"]
        fig = plt.figure(figsize = (1, 1), facecolor = "black")
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.fill(o[0, :] + b[2], o[1, :] + b[0], color = "white")
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        plt.savefig("mask.png", dpi = 224, facecolor = "black")
        plt.close()
        
        mask = Image.open("mask.png").convert("L")
        mask = mask.resize((224, 224))
        mask = np.array(mask, dtype = "uint8")
        mask[mask > 0] = 255
        
        x_temp.append(img)
        m_temp.append(mask)
        y_temp.append(label)

    #equal weight dataset by randomly deleting images
    #85~avg number of images per class 
    while len(x_temp) > 85:
        r = random.randrange(len(x_temp))
        del x_temp[r]
        del m_temp[r]
        del y_temp[r]
    x.extend(x_temp)
    m.extend(m_temp)
    y.extend(y_temp)

#detect images with no masks
no_masks = []
for i in range(len(m)):
  if m[i].any() == 0:
    no_masks.append(i)
print("no masks avaliable for {} images".format(len(no_masks)))

#delete images with no masks
for i in sorted(no_masks, reverse = True):
    del x[i]
    del m[i]
    del y[i]

x = np.array(x, dtype = "uint8")
m = np.array(m, dtype = "uint8")
y = np.array(y, dtype = "uint8")

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
m_train, m_test, _, _ = train_test_split(m, y, test_size = 0.25, random_state = 42)

print("{} train images".format(len(x_train)))
print("{} test images".format(len(x_test)))
classes = []
for c in y:
    classes.append(np.argmax(c))
counts = np.unique(classes, return_counts = True)[1]
print("max num images per class: {}".format(counts.max()))
print("min num images per class: {}".format(counts.min()))
print("avg num images per class: {:.4f}".format(np.mean(counts)))

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)
np.save("m_train.npy", m_train)
np.save("m_test.npy", m_test)
