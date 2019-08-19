"""
Train and save program.

This program carries out Deep Learning on an image set, using a Convolutional Neural Network (CNN).
The program carries out training and test predictions on the image dataset (The CIFAR-10 dataset)
located at: https://www.cs.toronto.edu/~kriz/cifar.html.
TensorFlow and Keras (Sequential model) used.
Parameters/Hyper-parameters of the CNN model can be adjusted for experimentation.

Run this program 1st. It will save the trained model and history in the same directory that this file is located.
Next run predictor.py, which will use x_test and y_test data. Three graphs will also be given, with
the last one, a bar graph displaying how many images were correctly categorised.

Current model setup output: 84% training acc., 129s 3ms/step - loss: 0.4461 - acc: 0.8478 - val_loss: 0.7714 - val_acc: 0.7636
"""

import numpy as np
import pickle
from tensorflow import keras
from keras.datasets import cifar10
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras import optimizers


# load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("y_train initial shape: ", y_train.shape)
print("y_test initial shape: ", y_test.shape)

# normalise input data
x_train = x_train/np.max(x_train)
x_test = x_test/np.max(x_test)

# Convert class vectors to binary class matrices.
num_classes = 10  # Number of image classes
y_train = keras.utils.to_categorical(y_train, num_classes)  # each row of array is 1-hot: 0 0...0 1 0 etc.
y_test = keras.utils.to_categorical(y_test, num_classes)  # each row of array is 1-hot: 0 0...0 1 0 etc.

print("x_train final shape: ", x_train.shape)
print("y_train final shape: ", y_train.shape)
print("x_test final shape: ", x_test.shape)
print("y_test final shape", y_test.shape)


def training():
    """train CNN model"""

    # create Sequential model and add layers
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=x_train.shape[1:],
                     padding='same', activation='relu', data_format='channels_last'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model, define optimiser, loss and metrics etc.
    optimiser = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0000006)
    model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])

    # tests a training condition for every epoch.
    # If a set amount of epochs elapses without showing improvement, then automatically stop the training.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # fit model.
    history = model.fit(x_train, y_train, batch_size=None, epochs=100, verbose=1, callbacks=[early_stop], validation_split=0.1,
              validation_data=None, shuffle=True, class_weight=None, sample_weight=None,
              initial_epoch=0, steps_per_epoch=None, validation_steps=None)

    # save the trained model
    model.save('imageTrainedModel.h5')  # saves the trained model'
    del model  # deletes any existing model

    # save model history:
    f = open('history.pckl', 'wb')
    pickle.dump(history.history, f)
    f.close()


training()
