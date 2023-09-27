from utils import load_csv_to_list
from pprint import pprint

import csv

keep_label_list = ['qiaonan', 'chenbin', 'chenwenhao', 'ganjiaqi', 
                   'nanjing', 'ruanyouxiang', 'xiehao', 'xushihao', 
                   'yangqingpeng', 'yangquanguo', 'yangxiaoyu']

def write_list_to_csv(data, filename):
    """
    Writes a list of dictionaries to a CSV file.

    Parameters:
    - data (list of dict): The list of dictionaries to be written to CSV.
    - filename (str): The name of the output CSV file.

    Returns:
    None
    """
    
    # Extract fieldnames (column headers) from the dictionary keys
    # Assumes that all dictionaries in the list have the same keys
    fieldnames = data[0].keys()

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the headers
        csvwriter.writeheader()
        
        # Write the rows
        for row in data:
            csvwriter.writerow(row)


def modify_labels(data: list, modify_label_dict: dict, del_label_list: list = []):
    """
    Modify the labels in the data list according to the label_dict.

    Parameters:
    - data (list): The list of dictionaries to be modified.
    - label_dict (dict): The label dictionary. The keys are the original labels, and the values are the new labels.
    - del_label_list (list): The list of labels to be deleted.
    """
    # Del all rows with the labels in del_label_list
    data = [row for row in data if (row['label'] not in del_label_list)]
    
    for row in data:
        if row['label'] not in modify_label_dict.keys():
            continue
        row['label'] = modify_label_dict[row['label']]
    return data


def get_all_unique_labels(data:list):
    """
    Get all unique labels in the data list.
    
    Returns:
    - labels (dict): The dictionary of labels and their counts.
    """
    labels = dict()
    for row in data:
        lb = row['label']
        if lb not in labels.keys():
            labels[lb] = 1
        else:
            labels[lb] += 1
    
    return labels

def del_rows(data:list, del_labels:list=None, keep_labels:list=None):
    """ Delete some rows in a list of dictionaries.

    Args:
        data (list): data
        del_labels (list, optional): If not None, del all rows with the labels in this list. Defaults to None.
        keep_labels (list, optional): If not None, keep all rows with the labels in this list, delete the others. Defaults to None.

    Returns:
        data: the modified data
    """
    del_or_keep = 'keep' if keep_labels is not None else 'del'
    
    if del_or_keep == 'keep':
        data = [row for row in data if (row['label'] in keep_labels)]
    else:
        data = [row for row in data if (row['label'] not in del_labels)]
    # for row in data:
    #     # if del_or_keep == 'del' and row['label'] in del_labels:
    #     #     dt.remove(row)
    #     #     continue
    #     # if del_or_keep == 'keep' and row['label'] not in keep_labels:
    #     #     dt.remove(row)
    #     #     continue
        
    #     data.remove(row)
        
        
    return data