import csv
import random
import os


def create_csv_file(file_name, example_num, sequence_len):
    """ Create a csv file for testing

    Args:
        file_name (str): the file name of the csv file
        example_num (int): the number of examples
        sequence_len (int): the length of "hold_time", "inter_time", "distance" sequences
    """
    # If the file exists, delete it
    if os.path.exists(file_name):
        os.remove(file_name)
    
    # write the data to a CSV file
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        # define the header row
        header = ["uuid", "label", "action", "pose", "usrn_len", "pswd_len", "total_time", "hold-time", "inter-time", "distance"]
        writer.writerow(header)
        
        data = []
        for i in range(example_num):
            # uuid is unique for each example
            uuid = i + 1
            
            # label refers to the user's identity
            usr_num = example_num // 11 + 1
            label = random.randint(1, usr_num)
            
            # usrn_len and psed is a random number from 8 to 15
            usrn_len = random.randint(8, 15)
            pswd_len = random.randint(8, 15)
            # total_time is a random number from 30 to 70
            total_time = random.randint(30, 70)
            
            gesture_type_map = [
                [1,2,3,4,5],
                [1,2,3],
                [1,2,3]
            ]
            action = random.randint(1, 3)
            pose = random.choice(gesture_type_map[action-1])
            
            # hold_time is a list of random numbers from 100 to 500, len is sequence_len
            hold_time = [random.randint(100, 500) for _ in range(sequence_len)]
            
            # inter_time is a list of random numbers from 100 to 500, len is sequence_len
            inter_time = [random.randint(100, 500) for _ in range(sequence_len)]

            
            # distance is a list of random numbers from 100 to 1000, len is sequence_len
            distance = [random.randint(100, 1000) for _ in range(sequence_len)]
            
            
            this_data = [uuid, label, action, pose, usrn_len, pswd_len, total_time,
                         hold_time, inter_time, distance]
            data.append(this_data)
        writer.writerows(data)

