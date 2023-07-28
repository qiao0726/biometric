import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import os
import yaml

# When data length is < 30, we will pad the data with 0s
FIXED_INPUT_LENGTH = 30

def encode_non_numeric(s):
    # Check if the string contains only numbers
    if s.isdigit():
        return s
    # Encode non-numeric characters using ASCII encoding
    encoded_chars = []
    for char in s:
        if char.isdigit():
            encoded_chars.append(char)
        else:
            encoded_chars.append(str(ord(char)))
    
    return ''.join(encoded_chars)


def load_xls_to_dict(file_path):
    if file_path[-4:] != '.xls' or os.path.isfile(file_path) == False:
        return None
    
    data = pd.read_excel(file_path)
    return data.to_dict()

def load_csv_to_list(file_path):
    # # If the file is not a csv file
    # if file_path[-4:] != '.csv' or os.path.isfile(file_path) == False:
    #     return None
    # data = list()
    # # Read file as a list by rows
    # with open(file_path, newline='') as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:
    #         data.append(row)
    # return data
    if file_path[-4:] != '.csv' or os.path.isfile(file_path) == False:
        return None
    data = pd.read_csv(file_path).to_dict()
    result = list()
    for row_num in range(len(data[next(iter(data))])):
        this_row = dict()
        for key in data.keys():
            this_row[key] = data[key][row_num]
        result.append(this_row)
    
    return result
    

def load_csv_to_dict(file_path):
    if file_path[-4:] != '.csv' or os.path.isfile(file_path) == False:
        return None
    data = pd.read_csv(file_path).to_dict()
    return data

def load_model_config_yaml(cfg_file_name='ts_model_config.yaml'):
    cfg_path = '/home/qn/biometric/config/' + cfg_file_name
    model_config = dict()
    with open(cfg_path, 'r') as file:
        if os.path.exists(cfg_path):
            model_config = yaml.load(file, Loader=yaml.FullLoader)
        else:
            raise Exception(f'No model_config.yaml found in {cfg_path}')
    return model_config

def load_training_config_yaml(cfg_path='/home/qn/biometric/config/training_config.yaml'):
    training_config = dict()
    with open(cfg_path, 'r') as file:
        if os.path.exists(cfg_path):
            training_config = yaml.load(file, Loader=yaml.FullLoader)
        else:
            raise Exception(f'No training_config.yaml found in {cfg_path}')
    return training_config

def load_test_config_yaml(cfg_path='/home/qn/biometric/config/test_config.yaml'):
    testing_config = dict()
    with open(cfg_path, 'r') as file:
        if os.path.exists(cfg_path):
            testing_config = yaml.load(file, Loader=yaml.FullLoader)
        else:
            raise Exception(f'No test_config.yaml found in {cfg_path}')
    return testing_config

def split_csv_file(csv_file_path, subset1_path, subset2_path, subset1_ratio=0.8):
    # Load the login.csv file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    # Split the DataFrame into two separate DataFrames
    df1, df2 = train_test_split(df, test_size=(1 - subset1_ratio))
    # Save the two DataFrames to separate CSV files
    df1.to_csv(subset1_path, index=False)
    df2.to_csv(subset2_path, index=False)


    