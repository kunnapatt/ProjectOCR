import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
import sys

np.set_printoptions(threshold=sys.maxsize)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train = datagen.flow_from_directory(
    directory='testdir/fold1/Train',
    color_mode='grayscale',
    # class_mode='binary',
    target_size=(60, 60),
    shuffle=True,
)

validation_train = datagen.flow_from_directory(
    directory='testdir/fold1/Validation',
    color_mode='grayscale',
    target_size=(60, 60),
    shuffle=True,
)

test_set = datagen.flow_from_directory(
    directory='testdir/fold1/Test',
    color_mode='grayscale',
    target_size=(60, 60),
    shuffle=True,
)

model = tf.keras.Sequential()

def conv2D(epoch):
    # model.add(tf.keras.layers.Flatten())

    # model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Conv2D(10, (10, 10), activation='relu', input_shape=(60, 60, 1)))

    # model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(10, (5, 5), activation='relu'))

    # model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(5, (3, 3), activation='relu'))

    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(70, activation='softmax'))

    optimizer_adam = tf.keras.optimizers.Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])

    with tf.device('/gpu:0'):
        history = model.fit_generator(generator=train, epochs=epoch, validation_data=validation_train)

    return history

def fullyconnected(epoch):
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(70, activation='relu'))
    
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    
    # model.add(tf.keras.layers.Dense(256, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dense(256, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(70, activation='softmax'))

    optimizer_adam = tf.keras.optimizers.Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])

    with tf.device('/gpu:0'):
        history = model.fit_generator(generator=train, epochs=epoch, validation_data=validation_train)

    return history

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    y_true = y_true.classes
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# history = fullyconnected(1000)
history = conv2D(1)

acctest = model.evaluate_generator(test_set)

model.save('../Model/model16-convolution2d.hdf5')

print("Test acc = ", acctest)

# print(history.history.keys())
# print(history.history['val_acc'])

# plt.plot(history.history['acc'])
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.show()

# plt.plot(history.history['val_acc'])
# plt.ylabel('val_acc')
# plt.xlabel('epoch')
# plt.show()

y_pred = model.predict_generator(test_set, 55)
# y_pred = np.argmax(y_pred, axis=1)
print('Confusion Matrix')
# print(confusion_matrix(test_set.classes, y_pred))
plot_confusion_matrix(test_set, y_pred, classes=test_set[0][1])
# print(len(y_pred))
# print(len(test_set[0][1]))
plt.show()
# print('Classification Report')
# target_class = []
# for i in range(70):
#     target_class.append(str(i+1))
# print(classification_report(test_set.classes, y_pred, target_names=target_class))