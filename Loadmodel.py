import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
import sys

np.set_printoptions(threshold=sys.maxsize)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_set = datagen.flow_from_directory(
    directory='../testdir/fold1/Train',
    color_mode='grayscale',
    target_size=(60, 60),
    shuffle=True 
)

validation_set = datagen.flow_from_directory(
    directory='../testdir/fold1/Validation',
    color_mode='grayscale',
    target_size=(60, 60),
    shuffle=True 
)

test_set = datagen.flow_from_directory(
    directory='../testdir/fold1/Test',
    color_mode='grayscale',
    target_size=(60, 60),
    shuffle=False 
)


model = tf.keras.models.load_model('../../Model/model18-convolution2d.hdf5')

y_pred = model.predict_generator(test_set)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)