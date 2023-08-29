import csv
from os import path

def create_save(name_file):
    if not exists('data/{}.csv'.format(name_file)):
         with open('data/{}.csv'.format(name_file), 'w', newline='') as file:
              writer = csv.writer(file)
              writer.writerow(["Hidden_Layer1", "HHidden_Layer2", "Learning_Rate","Batch_Size","Loss"])
              file.close()


def save_results():
    with open('data/training_FCNN_results.csv', 'a', newline='') as file:
              writer = csv.writer(file)
              writer.writerow([L1, L2, l_rate,bt_size,history_dict['loss'][epoc-1]])
    file.close()