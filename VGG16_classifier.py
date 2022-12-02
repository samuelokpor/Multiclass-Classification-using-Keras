import cv2
import os
import keras
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Flatten, Dense
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


#set seed
tf.random.set_seed(0)



IMG_PATH = os.path.join('inteva_fenlei_Images','item1','2022-11-25_10_12_05_139.jpg')

img = cv2.imread(IMG_PATH)
img = cv2.resize(img, (224, 224))
# cv2.imshow("Frame", img)
# cv2.waitKey(0)

#VGG16 model

model = keras.Sequential()

#block 1(conv layers)

model.add(Conv2D(64, kernel_size=(3,3), padding="same", activation="relu", input_shape=(224, 224, 3)))
model.add(Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#add feauture maps to see what model sees at this point

#block 2
model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#BLock 3
model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#Block 4
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#block 5
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#Top
model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dense(4096, activation="relu"))
model.add(Dense(6, activation="softmax"))

model.build()
model.summary()

#result
result = model.predict(np.array([img]))
print(model.summary())

#DIsplay Feature Map
# for i in range(64):
#     feature_img = feature_map[0, :, :, i]
#     ax = plt.subplot(8,8, i+1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.imshow(feature_img, cmap="gray")

# plt.show()
