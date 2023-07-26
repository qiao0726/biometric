from collections import OrderedDict
from torch import Tensor
import torch.nn as nn
from RNN import RNN
from GRU import SingleGRU
from FCNetwork import FCNetwork
from LSTM import LSTM
from factory import create_recognition_model, get_sensor_data_model, get_touchscreen_data_model
import torch

class GestureClassificationNetwork(nn.Module):
    def __init__(self, ts_data_network_name='RNN', sensor_data_network_name='RNN',
                 gesture_type_embedding_dim=8):
        super(GestureClassificationNetwork, self).__init__()
        self.touchscreen_data_network, out_dim1 = get_touchscreen_data_model(ts_data_network_name)
        self.sensor_data_network, out_dim2 = get_sensor_data_model(sensor_data_network_name)
        self.fc = nn.Linear(out_dim1 + out_dim2 + 2, 8)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, sensor_data, ts_data, total_time, usrname_pswd_len):
        ts_data_embeddings = self.touchscreen_data_network(ts_data)
        sensor_data_embeddings = self.sensor_data_network(sensor_data)
        embeddings = torch.cat((ts_data_embeddings, sensor_data_embeddings, total_time, usrname_pswd_len), dim=1)
        embeddings = self.fc(embeddings)
        cls_prob = self.softmax(embeddings)
        gesture_type_code = torch.argmax(cls_prob, dim=1)
        return gesture_type_code
        

class IDRecognitionNetwork(nn.Module):
    def __init__(self, ts_data_network_name='RNN', 
                 gesture_type_embedding_dim=8, out_feat_dim=64):
        super(IDRecognitionNetwork, self).__init__()
        # Choose from RNN, LSTM, GRU and Fully Connected Network
        self.touchscreen_data_network, out_dim = create_recognition_model(ts_data_network_name)
        self.gesture_type_embedding = nn.Embedding(8, gesture_type_embedding_dim)
        self.fc = nn.Linear(out_dim + gesture_type_embedding_dim + 2, out_feat_dim)
        
    def forward(self, ts_data, gst_type_code, total_time, usrname_pswd_len):
        ts_data_embeddings = self.touchscreen_data_network(ts_data)
        gst_type_embeddings = self.gesture_type_embedding(gst_type_code)
        embeddings = torch.cat((ts_data_embeddings, gst_type_embeddings, total_time, usrname_pswd_len), dim=1)
        id_embeddings = self.fc(embeddings)
        return id_embeddings
    
    
class BioMetricNetwork(nn.Module):
    def __init__(self, gesture_classfication_model:GestureClassificationNetwork, 
                 id_recognition_model:IDRecognitionNetwork):
        super(BioMetricNetwork, self).__init__()
        self.gst_cls_model = gesture_classfication_model
        self.id_recog_model = id_recognition_model
        
    def forward(self, sensor_data, ts_data, total_time, usrname_pswd_len):
        gesture_type_code = self.gst_cls_model(sensor_data, ts_data, total_time, usrname_pswd_len)
        id_embeddings = self.id_recog_model(ts_data, gesture_type_code, total_time, usrname_pswd_len)
        return id_embeddings
    
    def load_checkpoint(self, gst_cls_model_ckpt_path=None, id_recog_model_ckpt_path=None):
        if gst_cls_model_ckpt_path is not None:
            gst_cls_model_ckpt = torch.load(gst_cls_model_ckpt_path)
            self.gst_cls_model.load_state_dict(gst_cls_model_ckpt)
        if id_recog_model_ckpt_path is not None:
            id_recog_model_ckpt = torch.load(id_recog_model_ckpt_path)
            self.id_recog_model.load_state_dict(id_recog_model_ckpt)
        
        