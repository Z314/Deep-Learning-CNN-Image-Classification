"""
Test and predict on trained model (produced by train.py).

This program uses the trained model (imageTrainedModel.h5, history.pckl) that train.py produces.
The program uses x_test to make predictions and y_test to compare those predictions.
Dataset: (The CIFAR-10 dataset).
located at: https://www.cs.toronto.edu/~kriz/cifar.html.

Run this program 2nd (after train.py). Three graphs will be displayed upon execution of this file, with
the last one, a bar graph displaying how many images were correctly categorised in the test set.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import load_model


# load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalise data
x_train = x_train/np.max(x_train)
x_test = x_test/np.max(x_test)

print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape", y_test.shape)


# make predictions with trained model using x_test and y_test and then graph
def prediction():
    """loads saved model from file and test it on test data"""

    loaded_model = load_model('imageTrainedModel.h5')
    print(loaded_model.summary())

    # retrieve history also:
    f = open('history.pckl', 'rb')
    history = pickle.load(f)
    f.close()

    print(history.keys())
    print(history)

    epochs = len(history['loss'])  # length of the list stored at 'loss'
    # Plot losses for train and validation
    plt.figure()
    plt.title('Loss as training progresses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history['loss'],  label='Train Error')
    plt.plot(history['val_loss'], label='Val Error')
    plt.legend()
    plt.show()

    # Plot metrics
    plt.plot(history['acc'])  # use same metric that was used for training. 'history' is a dictionary.
    plt.title('Accuracy as training progresses')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    ymax = max(history['acc'])
    xpos = history['acc'].index(ymax)
    xmax = xpos
    plt.annotate('Maximum accuracy: %s' % round(ymax, 3),
                xy=(xmax, ymax), xycoords='data',
                xytext=(0.5, 0.5), textcoords='axes fraction',
                fontsize=12)
    plt.show()

    # make predictions using x_test
    test_y_predictions = loaded_model.predict(x_test, batch_size=None, verbose=1, steps=None)
    test_y_predictions = np.around(test_y_predictions, decimals=0)  # round to whole integers
    true_false_array = np.equal(y_test, test_y_predictions)  # test of equality.
    true_count = np.sum(true_false_array)  # number of correctly categorised images
    false_count = true_false_array.shape[0] - true_count  # number of images not correctly categorised

    # Plot predicted and actual image categories
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.title('Classification of Image Categories')
    plt.ylabel('Number of Images')
    plt.xlabel('Image Classification')
    label = ['Correct', 'Incorrect']
    index = np.arange(len(label))
    plt.xticks(index, label, fontsize=10, rotation=0)
    ax1.bar(index, [true_count, false_count])
    plt.show()


prediction()
