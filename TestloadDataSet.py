import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

#array[sample]

train = datagen.flow_from_directory(
    directory='../Dataset/Train/',
    color_mode='grayscale',
    # class_mode='binary',
    target_size=(60, 60),
)

# print('Batches train=%d' % (len(train)))

# print(arr)

# test = datagen.flow_from_directory(
#     directory='../Dataset/Test/',
#     color_mode='grayscale',
#     class_mode='categorical'
# )

# train = tf.keras.utils.normalize(train)

# plt.imshow(train[0][0][0][0][0], cmap=plt.cm.binary)
# plt.show()
# print(train[0])

# def fully_connected:
model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(70, activation='softmax'))

optimizer_adam = tf.keras.optimizers.Adam(lr=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])

# model.summary()
# with tf.device('/gpu:0'):
#     history = model.fit_generator(generator=train, epochs=1000)

history = model.fit_generator(generator=train, epochs=1000)

# print(history.history.keys())
# print(history.history['val_acc'])

plt.plot(history.history['acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# # plt.plot(history.history['loss'])
# # plt.ylabel('loss')
# # plt.xlabel('epoch')
# # plt.show()
# # plt.plot(history.history['acc'])
# # plt.plot(history.history['acc'])

# # print(datagen)
# # print(tf.VERSION)

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# import tensorflow as tf
# print(tf.VERSION)
