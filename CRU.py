import csv
from os import path

def write_to_csv(filename, row_number, data):
    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)

            # Check if the row number exists in the file
            if row_number <= len(rows):
                rows[row_number - 1] = data  # Overwrite the specified row
            else:
                while len(rows) < row_number:
                    rows.append([])  # Add empty rows till the desired row

                rows[row_number - 1] = data  # Overwrite the specified row

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        #print("Data has been written to the specified row successfully.")

    except FileNotFoundError:
        print("File not found.")

def create_save(name_file):
    if not exists('data/{}.csv'.format(name_file)):
         with open('data/{}.csv'.format(name_file), 'w', newline='') as file:
              writer = csv.writer(file)
              writer.writerow(["Hidden_Layer1", "Hidden_Layer2", "Learning_Rate","Batch_Size","Loss"])
              file.close()


def save_results():
    with open('data/training_FCNN_results.csv', 'a', newline='') as file:
              writer = csv.writer(file)
              writer.writerow([L1, L2, l_rate,bt_size,history_dict['loss'][epoc-1]])
    file.close()