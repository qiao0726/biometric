import torch.nn as nn
from .factory import get_sensor_data_model, get_touchscreen_data_model
import torch
import torch.nn.functional as F

class GestureClassificationNetwork(nn.Module):
    def __init__(self, ts_data_network_name='RNN', sensor_data_network_name='RNN', bidirectional=False):
        super(GestureClassificationNetwork, self).__init__()
        self.touchscreen_data_network, out_dim1 = get_touchscreen_data_model(ts_data_network_name, bidirectional=bidirectional)
        self.sensor_data_network, out_dim2 = get_sensor_data_model(sensor_data_network_name, bidirectional=bidirectional)
        self.fc = nn.Linear(out_dim1 + out_dim2 + 2, 8)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, sensor_data, ts_data, total_time, usrname_pswd_len):
        if isinstance(ts_data, list):
            # Stack tensors along new dimension
            ts_data = torch.stack(ts_data, dim=2)
        
        ts_data_embeddings = self.touchscreen_data_network(ts_data, None)
        sensor_data_embeddings = self.sensor_data_network(None, sensor_data)
        embeddings = torch.cat((ts_data_embeddings, sensor_data_embeddings, total_time, usrname_pswd_len), dim=1)
        embeddings = self.fc(embeddings)
        return embeddings
        

class IDRecognitionNetwork(nn.Module):
    def __init__(self, ts_data_network_name='RNN', 
                 gesture_type_embedding_dim=8, out_feat_dim=64, use_gesture_type=True,
                 bidirectional=False):
        super(IDRecognitionNetwork, self).__init__()
        # Choose from RNN, LSTM, GRU and Fully Connected Network
        self.touchscreen_data_network, out_dim = get_touchscreen_data_model(ts_data_network_name, bidirectional=bidirectional)
        self.use_gesture_type = use_gesture_type # Whether to use gesture type as an input
        if self.use_gesture_type:
            self.fc = nn.Linear(out_dim + gesture_type_embedding_dim + 2, out_feat_dim)
        else:
            self.fc = nn.Linear(out_dim + 2, out_feat_dim)
        
        
    def forward(self, ts_data, gst_type_code, total_time, usrname_pswd_len):
        if isinstance(ts_data, list):
            # Stack tensors along new dimension
            ts_data = torch.stack(ts_data, dim=2)
        
        ts_data_embeddings = self.touchscreen_data_network(ts_data, None)
        if self.use_gesture_type:
            embeddings = torch.cat((ts_data_embeddings, gst_type_code, total_time, usrname_pswd_len), dim=1)
        else:
            embeddings = torch.cat((ts_data_embeddings, total_time, usrname_pswd_len), dim=1)
        id_embeddings = self.fc(embeddings)
        return id_embeddings
    
    
class BioMetricNetwork(nn.Module):
    def __init__(self, gesture_classfication_model:GestureClassificationNetwork, 
                 id_recognition_model:IDRecognitionNetwork):
        super(BioMetricNetwork, self).__init__()
        self.gst_cls_model = gesture_classfication_model
        self.id_recog_model = id_recognition_model
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, sensor_data, ts_data, total_time, usrname_pswd_len):
        if isinstance(ts_data, list):
            # Stack tensors along new dimension
            ts_data = torch.stack(ts_data, dim=2)
        ts_data = ts_data.to(torch.device('cuda:7' if torch.cuda.is_available() else 'cpu'))
        
        gesture_type_embeddings = self.gst_cls_model(sensor_data, ts_data, total_time, usrname_pswd_len)
        
        cls_prob = self.softmax(gesture_type_embeddings)
        gst_type = torch.argmax(cls_prob, dim=1)
        # Convert index to one-hot encoding
        gst_type_one_hot = F.one_hot(gst_type, num_classes=cls_prob.shape[1])
        
        id_embeddings = self.id_recog_model(ts_data, gst_type_one_hot, total_time, usrname_pswd_len)
        return id_embeddings
    
    def load_checkpoint(self, gst_cls_model_ckpt_path=None, id_recog_model_ckpt_path=None, entire_model_ckpt_path=None):
        if gst_cls_model_ckpt_path is not None:
            gst_cls_model_ckpt = torch.load(gst_cls_model_ckpt_path)
            self.gst_cls_model.load_state_dict(gst_cls_model_ckpt)
        if id_recog_model_ckpt_path is not None:
            id_recog_model_ckpt = torch.load(id_recog_model_ckpt_path)
            self.id_recog_model.load_state_dict(id_recog_model_ckpt)
        if entire_model_ckpt_path is not None:
            entire_model_ckpt = torch.load(entire_model_ckpt_path)
            self.load_state_dict(entire_model_ckpt)
            

class NoGestureNetwork(nn.Module):
    def __init__(self, ts_data_network_name='RNN', sensor_data_network_name='RNN', out_feat_dim=64, bidirectional=False):
        super(NoGestureNetwork, self).__init__()
        # Choose from RNN, LSTM, GRU and Fully Connected Network
        self.touchscreen_data_network, out_dim1 = get_touchscreen_data_model(ts_data_network_name, bidirectional=bidirectional)
        self.sensor_data_network, out_dim2 = get_sensor_data_model(sensor_data_network_name, bidirectional=bidirectional)
        self.fc = nn.Linear(out_dim1 + out_dim2 + 2, out_feat_dim)
        
        
    def forward(self, sensor_data, ts_data, total_time, usrname_pswd_len):
        if isinstance(ts_data, list):
            # Stack tensors along new dimension
            ts_data = torch.stack(ts_data, dim=2)
        
        ts_data_embeddings = self.touchscreen_data_network(ts_data, None)
        sensor_data_embeddings = self.sensor_data_network(None, sensor_data)
        concat_data = torch.cat((ts_data_embeddings, sensor_data_embeddings, total_time, usrname_pswd_len), dim=1)
        id_embeddings = self.fc(concat_data)
        return id_embeddings
    
        
        