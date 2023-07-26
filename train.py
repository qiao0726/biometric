import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from dataset import TouchscreenDataset
from models.factory import create_recognition_model
from engines.training_engine import TripletTrainEngine

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--checkpoint_path', type=str, default='LSTM')
    parser.add_argument('--model_name', type=str, default='LSTM')
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

def get_dataloader(split='train', batch_size=128, data_path='data/touchscreen_data.csv'):
    dataset = TouchscreenDataset(csv_file_path=data_path)
    dataloder = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloder


def main(args):
    # model = create_model(args.model_name)
    # dataloader = get_dataloader(split='train', batch_size=args.batch_size, data_path=args.data_path)
    # train_engine = get_train_engine(model, dataloader, args.device, epochs=args.epochs, lr=args.lr,
    #                                 save_path=args.checkpoint_path, batch_size=args.batch_size)
    
    # train_engine.train(dataloader, model, optimizer, criterion, device)
    model = create_recognition_model(model_name=args.model_name)
    train_engine = TripletTrainEngine(model=model)
    train_engine.train()
    

if __name__ == '__main__':
    args = get_args()
    main(args)