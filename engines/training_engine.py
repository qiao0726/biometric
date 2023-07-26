from torch.utils.data import DataLoader
from dataset import TouchscreenDataset, TouchscreenSensorDataset
import torch
from utils import load_training_config_yaml
from losses.triplet_loss import OnlineTripletLoss
import logging
import time
from datetime import datetime
import os
import torch.utils.data as data_utils

train_cfg = load_training_config_yaml()

class TripletTrainEngine(object):
    def __init__(self, model, model_type='gesture_classification'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and train_cfg['Training']['use_cuda'] else 'cpu')
        
        # Split dataset into train and eval
        dataset = TouchscreenSensorDataset(csv_file_path=train_cfg['Training']['data_path'], sensor_data_folder_path=train_cfg['Training']['sensor_data_path'])
        train_size = int(train_cfg['Training']['train_set_ratio'] * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = data_utils.random_split(dataset, [train_size, test_size])
        
        self.train_loader = DataLoader(train_dataset, shuffle=False, batch_size=train_cfg['Training']['batch_size'])
        self.eval_loader = DataLoader(test_dataset, shuffle=False, batch_size=train_cfg['Training']['batch_size'])
        
        self.model = model.to(self.device)
        
        if train_cfg['Training']['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['Training']['lr'], 
                                              weight_decay=train_cfg['Training']['weight_decay'])
        elif train_cfg['Training']['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg['Training']['lr'], 
                                             weight_decay=train_cfg['Training']['weight_decay'])
        else:
            wrong_optimizer = train_cfg['Training']['optimizer']
            raise Exception(f'No optimizer named {wrong_optimizer}')
        #self.optimizer = self.optimizer.to(self.device)
        
        if train_cfg['Training']['loss_fn'] == 'TripletLoss':
            self.loss_fn = OnlineTripletLoss(margin=train_cfg['TripletLoss']['margin'], 
                                             batch_hard=train_cfg['TripletLoss']['batch_hard'], 
                                             squared=False)
        else:
            raise Exception(f'No loss function named {train_cfg["Training"]["loss_fn"]}')
        
        if train_cfg['Training']['lr_scheduler'] == 'StepLR':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                                step_size=train_cfg['Training']['step_size'],
                                                                gamma=train_cfg['Training']['gamma'])
        else:
            raise Exception(f'No lr scheduler named {train_cfg["Training"]["lr_scheduler"]}')
        
        self.epoch_num = train_cfg['Training']['epoch_num']
        self.log_interval = train_cfg['Training']['log_interval']
        self.eval_interval = train_cfg['Training']['eval_interval']
        self.best_acc = 0.0
        self.best_ckpt_name = ''
        
    def eval(self):
        self.model.eval()
        eval_stats = dict()
        eval_loss = 0.0
        correct_pairs = 0
        total_pairs = 0
        for batch_idx, batch in enumerate(self.eval_loader):
            sensor_data, ts_data, total_time, gesture_type, usrn_psrd_len, label = batch
            hold_time, inter_time, distance, speed = ts_data
            
            label = label.to(self.device)
            hold_time, inter_time = hold_time.to(self.device), inter_time.to(self.device)
            distance, speed = distance.to(self.device), speed.to(self.device)
            total_time, gesture_type = total_time.to(self.device), gesture_type.to(self.device)
            
            # embeddings.shape = (batch_size, embedding_size)
            embeddings = self.model(None, hold_time, inter_time, distance, speed, total_time, gesture_type)

            loss, batch_correct, batch_samples_num = self.loss_fn(embeddings, label)
            eval_loss += loss.item()
            correct_pairs += batch_correct
            total_pairs += batch_samples_num
        eval_stats['loss'] = eval_loss / (batch_idx+1)
        eval_stats['accuracy'] = correct_pairs / total_pairs
        return eval_stats
    
    def save_checkpoint(self, save_path, epoch, save_all=False, use_time=True):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_all:
            save_dict = {'epoch': epoch,
                         'model': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'lr_scheduler': self.lr_scheduler.state_dict()}
        else:
            save_dict = self.model.state_dict()
        if use_time:
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            ckpt_save_name = f'{save_path}/{self.model.__class__.__name__}_{current_time}.pth'
        else:
            ckpt_save_name = f'{save_path}/{self.model.__class__.__name__}.pth'
        torch.save(save_dict, ckpt_save_name)
        # Lastly saved checkpoint is the best checkpoint
        self.best_ckpt_name = ckpt_save_name.split('/')[-1]
        
    
    def train_one_epoch(self, current_epoch, total_epoch):
        train_epoch_stats = dict()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            self.model.train()
            sensor_data, ts_data, total_time, gesture_type, usrn_psrd_len, label = batch
            hold_time, inter_time, distance, speed = ts_data
            
            label =  label.to(self.device)
            hold_time, inter_time = hold_time.to(self.device), inter_time.to(self.device)
            distance, speed = distance.to(self.device), speed.to(self.device)
            total_time, gesture_type = total_time.to(self.device), gesture_type.to(self.device)
            
            # embeddings.shape = (batch_size, embedding_size)
            embeddings = self.model(None, hold_time, inter_time, 
                                    distance, speed, total_time, gesture_type)

            loss, correct, total = self.loss_fn(embeddings, label)
            epoch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            
            # Log the training information
            if batch_idx % self.log_interval == 0:
                print(f'[TRAIN]EPOCH: {current_epoch}/{total_epoch}, batch: {batch_idx}, loss: {loss.item():.4f}, lr: {self.lr_scheduler.get_last_lr()[0]:.4f}')
            
            # Evaluate the model, save the model and log the evaluation information
            if batch_idx % self.eval_interval == 0:
                eval_stats = self.eval()
                print(f'[EVAL]EPOCH: {current_epoch}/{total_epoch}, batch: {batch_idx}, loss: {eval_stats["loss"]:.4f}, accuracy: {eval_stats["accuracy"]:.4f}')
                if eval_stats['accuracy'] > self.best_acc:
                    self.best_acc = eval_stats['accuracy']
                    self.save_checkpoint(save_path=train_cfg['Training']['checkpoint_path'], epoch=current_epoch)
                    print(f'[SAVE]EPOCH: {current_epoch}/{total_epoch}, batch: {batch_idx}, best_acc: {self.best_acc}, best_ckpt: {self.best_ckpt_name}')

        epoch_loss /= (batch_idx+1)
        return epoch_loss
    
    def train(self):
        self.model.train()
        for epoch in range(self.epoch_num):
            start_time = time.time()
            epoch_loss = self.train_one_epoch(current_epoch=epoch, total_epoch=self.epoch_num)
            epoch_time = time.time() - start_time
            epoch_time = time.strftime('%H:%M:%S', time.gmtime(epoch_time))
            print(f'[TRAIN]EPOCH: {epoch}/{self.epoch_num}, Epoch Time: {epoch_time}, Epoch Loss: {epoch_loss:.4f}')
            
            # Eval model at the end of each epoch
            eval_stats = self.eval()
            print(f'[EVAL]EPOCH: {epoch}/{self.epoch_num}, loss: {eval_stats["loss"]:.4f}, accuracy: {eval_stats["accuracy"]:.4f}')
            if eval_stats['accuracy'] > self.best_acc:
                self.best_acc = eval_stats['accuracy']
                self.save_checkpoint(save_path=train_cfg['Training']['checkpoint_path'], epoch=epoch)
                print(f'[SAVE]EPOCH: {epoch}/{self.epoch_num}, best_acc: {self.best_acc}, best_ckpt: {self.best_ckpt_name}')
            
        print(f'[DONE]Best_acc: {self.best_acc}, Best_ckpt: {self.best_ckpt_name}')
        return
    
    
        
        