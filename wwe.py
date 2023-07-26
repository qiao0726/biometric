from dataset import SensorDataset
from torch.utils.data import DataLoader
import pandas as pd
from pprint import pprint

sensor_ds = SensorDataset(csv_file_path=r'/home/wcy/shengwutanzhen/data/login.csv', 
                          sensor_data_folder_path=r'/home/wcy/shengwutanzhen/data/sensor')

dl = DataLoader(sensor_ds, batch_size=3, shuffle=False)

for batch_idx, (sensor_data, label, total_time, gesture_type) in enumerate(dl):
    # sensor_data.shape = (batch_size, sequence_length, input_size), input_size = 6 in this case
    print(sensor_data.shape)
    print(label)
    print(total_time)
    print(gesture_type)
    

