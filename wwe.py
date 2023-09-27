from csv_utils import write_list_to_csv, modify_labels, get_all_unique_labels, del_rows
from utils import load_csv_to_list
from pprint import pprint

csv_file_path = r'/home/qn/biometric/data/0921new_data/login.csv'
data, _ = load_csv_to_list(csv_file_path)

pprint(get_all_unique_labels(data))

keep_label_list = ['qiaonan', 'chenbin', 'chenwenhao', 'ganjiaqi', 
                   'nanjing', 'ruanyouxiang', 'xiehao', 'xushihao', 
                   'yangqingpeng', 'yangquanguo', 'yangxiaoyu']

data = del_rows(data=data, keep_labels=keep_label_list)

write_list_to_csv(data, r'/home/qn/biometric/data/0921new_data/login2.csv')