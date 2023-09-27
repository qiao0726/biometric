# from engines.test_engine import TestEngine
import argparse
from models.networks import GestureClassificationNetwork, IDRecognitionNetwork, BioMetricNetwork, NoGestureNetwork
import torch
from dataset import TouchscreenSensorDataset
from utils import load_csv_to_list
from torch.utils.data import DataLoader
from losses.triplet_loss import get_pairwise_distances
from visualization import visualize3d_color
from pprint import pprint


def create_test_model(sensor_model_name, ts_model_name, model_type, use_gesture_type, out_feat_dim, ckpt):
    if model_type == 'both':
        model = BioMetricNetwork(gesture_classfication_model=GestureClassificationNetwork(ts_data_network_name=ts_model_name,
                                                                                          sensor_data_network_name=sensor_model_name),
                                 id_recognition_model=IDRecognitionNetwork(ts_data_network_name=ts_model_name, gesture_type_embedding_dim=8, 
                                                                           out_feat_dim=out_feat_dim, use_gesture_type=True))
    elif model_type == 'gesture_only':
        model = GestureClassificationNetwork(ts_data_network_name=ts_model_name, 
                                             sensor_data_network_name=sensor_model_name)
    elif model_type == 'id_only':
        model = IDRecognitionNetwork(ts_data_network_name=ts_model_name, gesture_type_embedding_dim=8, 
                                    out_feat_dim=out_feat_dim, use_gesture_type=use_gesture_type)
        
    elif model_type == 'no_gesture':
        model = NoGestureNetwork(ts_data_network_name=ts_model_name, 
                                 sensor_data_network_name=sensor_model_name,
                                 out_feat_dim=out_feat_dim)
        
    else:
        raise Exception(f'No model type named {model_type}')
    
    state_dict = torch.load(ckpt)
    model.load_state_dict(state_dict)
    
    return model

def create_test_loader(testset, sensor_data_folder_path):
    dataset = TouchscreenSensorDataset(csv_file_path=testset, sensor_data_folder_path=sensor_data_folder_path)
    test_loader = DataLoader(dataset, shuffle=True, batch_size=32)
    return test_loader


def main(args):
    model = create_test_model(args.sensor_model_name, args.ts_model_name, args.model_type, 
                              args.use_gesture_type, args.out_feat_dim, args.ckpt)
    test_loader = create_test_loader(testset=r'data/login.csv', sensor_data_folder_path=r'/home/qn/biometric/data/sensor')
    device = torch.device('cuda:1' if False else 'cpu')
    model = model.to(device)
    model.eval()
    
    # label_dict = load_csv_to_list(r'data/login.csv')
    # pprint(label_dict)
    total_batch=0
    
    
    for batch_idx, batch in enumerate(test_loader):
        sensor_data, ts_data, total_time, gesture_type, usrn_pswd_len, label = batch
        total_time, usrn_pswd_len, label = total_time.unsqueeze(1), usrn_pswd_len.unsqueeze(1), label.unsqueeze(1)
        sensor_data = sensor_data.to(device)
        for data in ts_data:
            data = data.to(device)
        total_time, usrn_pswd_len = total_time.to(device), usrn_pswd_len.to(device)
        gesture_type, label = gesture_type.to(device), label.to(device)
        
        embeddings = model(sensor_data, ts_data, total_time, usrn_pswd_len)
        # pairwise_dist = get_pairwise_distances(embeddings)
        # pairwise_dist = pairwise_dist.float()
        
        visualize3d_color(embeddings.cpu(), save_path=f'{batch_idx}.png', labels=label.cpu())
        
        
        # # The 0s in pairwise_distances means the anchor itself, change it to inf
        # # Replace all 0 values with inf
        # pairwise_dist = torch.where(pairwise_dist == 0.0, torch.tensor(float('inf')).double(), pairwise_dist)
        # # Get the minimum value of the last row
        # min_value, _ = torch.min(pairwise_dist[-1], dim=0)
        # min_value = float(min_value.item())
        
        # print(f'batch_idx:{batch_idx}: min_value:{min_value}')
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--sensor_model_name', type=str, help='sensor model name')
    parser.add_argument('--ts_model_name', type=str, help='touchscreen model name')
    parser.add_argument('--model_type', type=str, help='Model name')
    parser.add_argument('--use_gesture_type', type=bool, help='use gesture type or not')
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--out_feat_dim', type=int, help='output feature dimension')
    parser.add_argument('--ckpt', type=str, help='Model checkpoint path')
    parser.add_argument('--testset', type=str, help='Testset csv file path')
    parser.add_argument('--dist_fn', type=str, default='euclidean', help='Choose from "cosine" and "euclidean"')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for cosine distance')
    args = parser.parse_args()
    main(args)