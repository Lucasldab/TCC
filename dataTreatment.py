def clean_data(data):
    clean_data = data[data["Name"] != 'Adafactor']

    clean_data["Name"] = clean_data["Name"].replace(['SGD'],0.1)
    clean_data["Name"] = clean_data["Name"].replace(['Adam'],0.3)
    clean_data["Name"] = clean_data["Name"].replace(['RMSprop'],0.2)
    clean_data["Name"] = clean_data["Name"].replace(['AdamW'],0.4)
    clean_data["Name"] = clean_data["Name"].replace(['Adadelta'],0.5)
    clean_data["Name"] = clean_data["Name"].replace(['Adagrad'],0.6)
    clean_data["Name"] = clean_data["Name"].replace(['Adamax'],0.7)
    clean_data["Name"] = clean_data["Name"].replace(['Nadam'],0.8)
    clean_data["Name"] = clean_data["Name"].replace(['Ftrl'],0.9)

    clean_data.to_csv('data/training_CNN_results_v3-1.csv', index=False)
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

#def smallest_loss_train(smallest_loss_local,data_only,loss_data):
#    smallest = np.append(data_only[min_loc],loss_data[min_loc])
#    return smallest
