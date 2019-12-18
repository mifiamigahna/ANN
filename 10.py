from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import keyboard as kb

from keras.datasets import fashion_mnist


def show_samples(rows, columns):
    fig, images = plt.subplots(rows, columns)
    fig.tight_layout()
    for image in images.ravel():
        r = np.random.randint(0, 50000)
        image.imshow(Xtrain[r], cmap='Greys')
        image.axis('off')
        image.set_title(clothing[Ytrain[r]])


def load_data():
    (XtrainMat, Ytrain), (XtestMat, Ytest) = fashion_mnist.load_data()

    n_train = len(Ytrain)
    n_test = len(Ytest)

    p_train = np.random.permutation(n_train)
    p_test = np.random.permutation(n_test)

    XtrainMat, Ytrain = XtrainMat[p_train] / 255,  Ytrain[p_train]
    XtestMat, Ytest = XtestMat[p_test] / 255, Ytest[p_test]

    Xtrain = np.array([image.flatten() for image in XtrainMat])
    Xtest = np.array([image.flatten() for image in XtestMat])

    Xtrain = np.concatenate((np.ones((n_train, 1)), Xtrain), axis=1)
    Xtest = np.concatenate((np.ones((n_test, 1)), Xtest), axis=1)

    return Xtrain, Ytrain, Xtest, Ytest, XtrainMat, XtestMat


def check_class(class_index):
    Yhat = model.predict_classes(Xtest, batch_size=1)
    classified = np.nonzero(Yhat == class_index)[0]
    for index in classified:
        ax = plt.axes()
        ax.imshow(XtestMat[index], cmap='Greys')
        ax.axis('off')
        ax.set_title(clothing[Ytest[index]])
        plt.pause(0.01)
        if Ytest[index] != class_index:
            kb.wait('enter')


clothing = {0: 'T-shirt/Top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
            4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}

Xtrain, Ytrain, Xtest, Ytest, XtrainMat, XtestMat = load_data()

# #!DNN
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=785))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.05)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy',
              metrics=['acc'])

train_history_DNN = model.fit(
    Xtrain[:10000], Ytrain[:10000], batch_size=1000, epochs=3)
accuracy_DNN = model.test_on_batch(Xtest, Ytest)[1]


#!CNN
model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.05)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy',
              metrics=['acc'])

train_history_CNN = model.fit(XtrainMat[:10000].reshape(
    (-1, 28, 28, 1)), Ytrain[:10000], batch_size=1000, epochs=3)
accuracy_CNN = model.test_on_batch(XtestMat.reshape((-1, 28, 28, 1)), Ytest)[1]


print("CNN accuracy on the train set :",
      train_history_DNN.history['acc'][-1])
print("CNN accuracy on the test set :", accuracy_DNN)

print("DNN accuracy on the train set :", train_history_CNN.history['acc'][-1])
print("DNN accuracy on the test set :", accuracy_CNN)

# check_class(9)
