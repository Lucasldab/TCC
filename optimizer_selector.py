from tensorflow import keras
from keras import optimizers
import random

def defining_optimizer(x):
    l_rate = random.uniform(0.0001, 0.01)
    if (x <= 100):
        optimizer = optimizers.SGD(learning_rate=l_rate)
    elif (x <= 200):
        optimizer = optimizers.RMSprop(learning_rate=l_rate)
    elif (x <= 300):
        optimizer = optimizers.Adam(learning_rate=l_rate)
    elif (x <= 400):
        optimizer = optimizers.AdamW(learning_rate=l_rate)
    elif (x <= 500):
        optimizer = optimizers.Adadelta(learning_rate=l_rate)
    elif (x <= 600):
        optimizer = optimizers.Adagrad(learning_rate=l_rate)
    elif (x <= 700):
        optimizer = optimizers.Adamax(learning_rate=l_rate)
    elif (x <= 800):
        optimizer = optimizers.Adafactor(learning_rate=l_rate)
    elif (x <= 900):
        optimizer = optimizers.Nadam(learning_rate=l_rate)
    elif (x <= 1000):
        optimizer = optimizers.Ftrl(learning_rate=l_rate)
    return optimizer, l_rate


def randomize_optimizer(optimizer_number,l_rate):
    if (optimizer_number == 1):
        optimizer = optimizers.SGD(learning_rate=l_rate)
        name = optimizers.SGD.__name__
    elif (optimizer_number == 2):
        optimizer = optimizers.RMSprop(learning_rate=l_rate)
        name = optimizers.RMSprop.__name__
    elif (optimizer_number == 3):
        optimizer = optimizers.Adam(learning_rate=l_rate)
        name = optimizers.Adam.__name__
    elif (optimizer_number == 4):
        optimizer = optimizers.AdamW(learning_rate=l_rate)
        name = optimizers.AdamW.__name__
    elif (optimizer_number == 5):
        optimizer = optimizers.Adadelta(learning_rate=l_rate)
        name = optimizers.Adadelta.__name__
    elif (optimizer_number == 6):
        optimizer = optimizers.Adagrad(learning_rate=l_rate)
        name = optimizers.Adagrad.__name__
    elif (optimizer_number == 7):
        optimizer = optimizers.Adamax(learning_rate=l_rate)
        name = optimizers.Adamax.__name__
    elif (optimizer_number == 8):
        optimizer = optimizers.Nadam(learning_rate=l_rate)
        name = optimizers.Nadam.__name__
    elif (optimizer_number == 9):
        optimizer = optimizers.Ftrl(learning_rate=l_rate)
        name = optimizers.Ftrl.__name__
    return optimizer, name

def defining_optimizer_byName(optimizer,l_rate):
    if (optimizer == 'SGD'):
        optimizer = optimizers.SGD(learning_rate=l_rate)
        name = optimizers.SGD.__name__
    elif (optimizer == 'RMSprop'):
        optimizer = optimizers.RMSprop(learning_rate=l_rate)
        name = optimizers.RMSprop.__name__
    elif (optimizer == 'Adam'):
        optimizer = optimizers.Adam(learning_rate=l_rate)
        name = optimizers.Adam.__name__
    elif (optimizer == 'AdamW'):
        optimizer = optimizers.AdamW(learning_rate=l_rate)
        name = optimizers.AdamW.__name__
    elif (optimizer == 'Adadelta'):
        optimizer = optimizers.Adadelta(learning_rate=l_rate)
        name = optimizers.Adadelta.__name__
    elif (optimizer == 'Adagrad'):
        optimizer = optimizers.Adagrad(learning_rate=l_rate)
        name = optimizers.Adagrad.__name__
    elif (optimizer == 'Adamax'):
        optimizer = optimizers.Adamax(learning_rate=l_rate)
        name = optimizers.Adamax.__name__
    elif (optimizer == 'Nadam'):
        optimizer = optimizers.Nadam(learning_rate=l_rate)
        name = optimizers.Nadam.__name__
    elif (optimizer == 'Ftrl'):
        optimizer = optimizers.Ftrl(learning_rate=l_rate)
        name = optimizers.Ftrl.__name__
    return optimizer, name