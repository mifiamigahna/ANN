import numpy as np
import matplotlib.pyplot as plt
import keyboard as kb
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import SGD, Adam

from keras.datasets import fashion_mnist


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


def build_model_1(lr=0.001):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same',
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=lr)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model


def build_model_2(lr=0.001):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same',
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model


def train_model(model, Xtrain, Ytrain, bs=10, e=5):
    train_hist = model.fit(Xtrain, Ytrain, batch_size=bs, epochs=e)
    return (train_hist.history['loss'], train_hist.history['acc'])


def test_model(model, Xtest, Ytest):
    loss, acc = model.test_on_batch(Xtest, Ytest)
    return (loss, acc)


def test_lrs(Xtrain, Ytrain, lrs):
    loss = []
    accuracy = []

    # create figure
    fig, plots = plt.subplots(1, len(lrs))
    fig.tight_layout()

    # train models with lrs
    for model, lr in enumerate(lrs):
        print('\nLearning Rate:', lr)
        new_loss, new_acc = train_model(build_model_1(lr), Xtrain, Ytrain)

        loss.append(new_loss)
        accuracy.append(new_acc)

        # plot loss & accuracy
        plots.ravel()[model].set_title(str(lr))
        plots.ravel()[model].plot(loss[model], c='R')
        plots.ravel()[model].plot(accuracy[model], c='G')

    plt.show(fig)


def test_train_size(Xtrain, Ytrain, Xtest, Ytest, train_sizes):

    loss = [[], []]
    accuracy = [[], []]

    for train_size in train_sizes:

        # (re-)build models
        models = [build_model_1(), build_model_2()]

        # train & test models
        print('\nTraining Set Size:', train_size)
        for n, model in enumerate(models):

            print(f"\nModel: {n + 1}")
            train_model(model, Xtrain[:train_size], Ytrain[:train_size], e=10)
            new_loss, new_acc = test_model(model, Xtest, Ytest)
            loss[n].append(new_loss)
            accuracy[n].append(new_acc)

    # create figure
    fig, plots = plt.subplots(1, 2)
    fig.tight_layout()

    # plot loss & accuracy
    for model in range(2):
        plots.ravel()[model].set_title(f"Model {model + 1}")
        plots.ravel()[model].set_ylim(0, 1)
        plots.ravel()[model].plot(train_sizes, loss[model], c='R')
        plots.ravel()[model].plot(train_sizes, accuracy[model], c='G')

    plt.show()


lrs = [1, 0.1, 0.01, 0.001, 0.0001]
train_sizes = [500, 2500, 15000, 30000]
Xtrain, Ytrain, Xtest, Ytest, XtrainMat, XtestMat = load_data()

# ----- 1 -----
# test_lrs(XtrainMat[:20000].reshape((-1, 28, 28, 1)), Ytrain[:20000], lrs)

# ----- 2 -----
# test_train_size(XtrainMat.reshape((-1, 28, 28, 1)), Ytrain,
#                 XtestMat.reshape((-1, 28, 28, 1)), Ytest, train_sizes)
