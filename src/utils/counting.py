import os
import csv

# Function to count lines in a CSV file
def count_lines_in_csv(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        line_count = sum(1 for row in reader)
    return line_count

# Directory containing the CSV files
directory = r'D:\Projetos\PesquisaArtigo\TCC_Hyperparameters Optimization\data\random'  # Replace with the path to your folder

# Iterate through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        csv_file = os.path.join(directory, filename)
        line_count = count_lines_in_csv(csv_file)
        print(f"{filename}: {line_count} lines")