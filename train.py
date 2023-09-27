import argparse
from engines.training_engine import TripletTrainEngine
from models.networks import GestureClassificationNetwork, IDRecognitionNetwork, BioMetricNetwork, NoGestureNetwork
import torch.nn as nn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--model_type', type=str, default='id_only') # Choose from 'both', 'gesture_only' and 'id_only'
    parser.add_argument('--use_gesture_type', type=bool, default=True) # Whether to use gesture type as an input
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--out_feat_dim', type=int, default=64)
    parser.add_argument('--sensor_model_name', type=str, default='LSTM')
    parser.add_argument('--ts_model_name', type=str, default='LSTM')
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def main(args):
    if args.model_type == 'both':
        model = BioMetricNetwork(gesture_classfication_model=GestureClassificationNetwork(ts_data_network_name=args.ts_model_name,
                                                                                          sensor_data_network_name=args.sensor_model_name,
                                                                                          bidirectional=args.bidirectional),
                                 id_recognition_model=IDRecognitionNetwork(ts_data_network_name=args.ts_model_name, gesture_type_embedding_dim=8, 
                                                                           out_feat_dim=args.out_feat_dim, use_gesture_type=True,
                                                                           bidirectional=args.bidirectional))
        if args.checkpoint_path:
            model.load_checkpoint(entire_model_ckpt_path=args.checkpoint_path)
            print('Loaded checkpoint')
        
        train_engine = TripletTrainEngine(model=model, model_type=args.model_type, 
                                          sensor_model_name=args.ts_model_name, 
                                          ts_model_name=args.sensor_model_name)
    elif args.model_type == 'gesture_only':
        model = GestureClassificationNetwork(ts_data_network_name=args.ts_model_name, 
                                             sensor_data_network_name=args.sensor_model_name,
                                             bidirectional=args.bidirectional)
        train_engine = TripletTrainEngine(model=model, model_type=args.model_type, 
                                          sensor_model_name=args.ts_model_name, 
                                          ts_model_name=args.sensor_model_name)
    elif args.model_type == 'id_only':
        model = IDRecognitionNetwork(ts_data_network_name=args.ts_model_name, gesture_type_embedding_dim=8, 
                                    out_feat_dim=args.out_feat_dim, use_gesture_type=args.use_gesture_type,
                                    bidirectional=args.bidirectional)
        train_engine = TripletTrainEngine(model=model, model_type=args.model_type, 
                                          sensor_model_name=args.ts_model_name, 
                                          ts_model_name=args.sensor_model_name)
        
    elif args.model_type == 'no_gesture':
        model = NoGestureNetwork(ts_data_network_name=args.ts_model_name, 
                                 sensor_data_network_name=args.sensor_model_name,
                                 out_feat_dim=args.out_feat_dim,
                                 bidirectional=args.bidirectional)
        train_engine = TripletTrainEngine(model=model, model_type=args.model_type, 
                                          sensor_model_name=args.ts_model_name, 
                                          ts_model_name=args.sensor_model_name)
        
    else:
        raise Exception(f'No model type named {args.model_type}')
    
    
    train_engine.train()
    

if __name__ == '__main__':
    args = get_args()
    main(args)