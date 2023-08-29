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


def randomize_optimizer():
    optim_rand = random.randint(1,10)
    l_rate = random.uniform(0.0001, 0.01)
    if (optim_rand == 1):
        optimizer = optimizers.SGD(learning_rate=l_rate)
        name = optimizers.SGD.__name__
    elif (optim_rand == 2):
        optimizer = optimizers.RMSprop(learning_rate=l_rate)
        name = optimizers.RMSprop.__name__
    elif (optim_rand == 3):
        optimizer = optimizers.Adam(learning_rate=l_rate)
        name = optimizers.Adam.__name__
    elif (optim_rand == 4):
        optimizer = optimizers.AdamW(learning_rate=l_rate)
        name = optimizers.AdamW.__name__
    elif (optim_rand == 5):
        optimizer = optimizers.Adadelta(learning_rate=l_rate)
        name = optimizers.Adadelta.__name__
    elif (optim_rand == 6):
        optimizer = optimizers.Adagrad(learning_rate=l_rate)
        name = optimizers.Adagrad.__name__
    elif (optim_rand == 7):
        optimizer = optimizers.Adamax(learning_rate=l_rate)
        name = optimizers.Adamax.__name__
    elif (optim_rand == 8):
        optimizer = optimizers.Adafactor(learning_rate=l_rate)
        name = optimizers.Adafactor.__name__
    elif (optim_rand == 9):
        optimizer = optimizers.Nadam(learning_rate=l_rate)
        name = optimizers.Nadam.__name__
    elif (optim_rand == 10):
        optimizer = optimizers.Ftrl(learning_rate=l_rate)
        name = optimizers.Ftrl.__name__
    return optimizer, l_rate, name