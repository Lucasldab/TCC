import numpy as np

def clean_data(data):
    clean_data = data[data["Name"] != 'Adafactor']

    clean_data["Name"] = clean_data["Name"].replace(['SGD'],1.0)
    clean_data["Name"] = clean_data["Name"].replace(['Adam'],2.0)
    clean_data["Name"] = clean_data["Name"].replace(['RMSprop'],3.0)
    clean_data["Name"] = clean_data["Name"].replace(['AdamW'],4.0)
    clean_data["Name"] = clean_data["Name"].replace(['Adadelta'],5.0)
    clean_data["Name"] = clean_data["Name"].replace(['Adagrad'],6.0)
    clean_data["Name"] = clean_data["Name"].replace(['Adamax'],7.0)
    clean_data["Name"] = clean_data["Name"].replace(['Nadam'],8.0)
    clean_data["Name"] = clean_data["Name"].replace(['Ftrl'],9.0)

    #clean_data.to_csv('trainings/training_CNN_results_v3-1.csv', index=False)
    clean_data.astype('float32')
    return clean_data

def divide_samplings(clean_data):
    half_data = clean_data.sample(frac = 0.5)
    other_half_data = clean_data.drop(half_data.index)
    np.set_printoptions(suppress=True)
    other_half_data = other_half_data.to_numpy()

    return half_data,other_half_data

def data_from_loss(half_data):
    loss_data = half_data["Loss"]
    loss_data.astype('float32')
    loss_data = loss_data.to_numpy()
    smallest_loss_local = np.argmin(loss_data)
    
    half_data = half_data.drop('Loss', axis=1)
    data_only = np.asarray(half_data)

    return loss_data,data_only,smallest_loss_local

def decimalToName(optimizer):
    if (optimizer == 1):
        optimizer = 'SGD'
    elif (optimizer == 2):
        optimizer = 'RMSprop'
    elif (optimizer == 3):
        optimizer = 'Adam'
    elif (optimizer == 4):
        optimizer = 'AdamW'
    elif (optimizer == 5):
        optimizer = 'Adadelta'
    elif (optimizer == 6):
        optimizer = 'Adagrad'
    elif (optimizer == 7):
        optimizer = 'Adamax'
    elif (optimizer == 8):
        optimizer = 'Nadam'
    elif (optimizer == 9):
        optimizer = 'Ftrl'
    return optimizer