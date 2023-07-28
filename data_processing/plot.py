import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def read_csv_column(csv_file_path, header_name, step=10):
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        column = []
        for i, row in enumerate(reader):
            if i % step == 0:
                column.append(round(float(row[header_name]), 4))
    return column

# def plotCurve(csv_file_path, save_path, col, step=10):
#     plt.clf()
#     y_data = read_csv_column(csv_file_path, col, step=step)
#     x_data = np.linspace(0, len(y_data) * 0.1, len(y_data))

    
#     plt.plot(x_data, y_data)

#     # Add labels and title
#     plt.title(f'{col} Curve')
#     plt.xlabel('Time(s)')
#     plt.ylabel(col)
    
#     # Save as PNG file
#     plt.savefig(save_path)
#     return
login_csv_file_path = r'/home/wcy/shengwutanzhen/data/login.csv'
csv_file_path = r'/home/qn/biometric/data/sensor/dde110af2b13456c9ed917f4eba173c7.csv'
save_path = os.path.join(r'/home/qn/biometric/data_processing', csv_file_path.split('/')[-1].split('.')[0] + '.png')

def plotCurves(csv_file_path, save_path, col_names, step=10):
    plt.clf()
    fig, ax = plt.subplots()
    for col_name in col_names:
        y_data = read_csv_column(csv_file_path, col_name, step=step)
        x_data = list(range(0, len(y_data)))
        ax.plot(x_data, y_data, label=col_name)
    ax.legend()
    plt.savefig(save_path)



if __name__ == '__main__':
    plotCurves(csv_file_path, save_path, step=1, col_names=
               ('accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ'))
