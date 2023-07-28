import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from dataset import TouchscreenDataset, TouchscreenSensorDataset
from engines.training_engine import TripletTrainEngine
from models.networks import GestureClassificationNetwork, IDRecognitionNetwork, BioMetricNetwork

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--model_type', type=str, default='id_only') # Choose from 'both', 'gesture_only' and 'id_only'
    parser.add_argument('--checkpoint_path', type=str, default='LSTM')
    parser.add_argument('--out_feat_dim', type=int, default=64)
    parser.add_argument('--sensor_model_name', type=str, default='LSTM')
    parser.add_argument('--ts_model_name', type=str, default='LSTM')
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def main(args):
    if args.model_type == 'both':
        model = BioMetricNetwork(ts_data_network_name=args.ts_model_name, gesture_type_embedding_dim=8, 
                                    out_feat_dim=args.out_feat_dim)
        train_engine = TripletTrainEngine(model=model, model_type=args.model_type, model_name=args.ts_model_name + '_' + args.sensor_model_name)
        train_engine.train()
    elif args.model_type == 'gesture_only':
        model = GestureClassificationNetwork(ts_data_network_name=args.ts_model_name, sensor_data_network_name=args.sensor_model_name)
        train_engine = TripletTrainEngine(model=model, model_type=args.model_type, model_name=args.ts_model_name + '_' + args.sensor_model_name)
        train_engine.train()
    elif args.model_type == 'id_only':
        model = IDRecognitionNetwork(ts_data_network_name=args.ts_model_name, gesture_type_embedding_dim=8, 
                                    out_feat_dim=args.out_feat_dim)
        train_engine = TripletTrainEngine(model=model, model_type=args.model_type, model_name=args.ts_model_name)
        train_engine.train()
    else:
        raise Exception(f'No model type named {args.model_type}')
    

if __name__ == '__main__':
    args = get_args()
    main(args)