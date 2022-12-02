import tensorflow as tf
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
from VGG16_classifier import model



#avoid oom errors limiting gpu consumption
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

##load dataset
data = tf.keras.utils.image_dataset_from_directory('inteva_fenlei_images', image_size=(224,224))

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


##display data class
# fig, ax=plt.subplots(ncols=4, figsize=(20,20))
for img in enumerate(batch[0][:4]):
    # ax[idx].imshow(img.astype(int))
    # ax[idx].title.set_text(batch[1][idx])
    cv2.imshow("images", img.astype(int))
    cv2.waitKey(1)

if y> 0.5:
    pass
elif y< 0.5:

else