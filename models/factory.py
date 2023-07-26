from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .LSTM import LSTM
from .FCNetwork import FCNetwork
from .GRU import SingleGRU, HierGRU
from .RNN import RNN
import yaml
import os
from utils import load_model_config_yaml, load_test_config_yaml



def create_gesture_model(model_name):
    model_config = load_model_config_yaml()
    if model_name in model_config.keys():
        cfg = model_config[model_name]
    else:
        raise Exception(f'No model named {model_name} in model_config.yaml')
    
    if model_name == 'LSTM':
        return LSTM(input_size=cfg['input_size'], hidden_size=cfg['hidden_size'], 
                    num_layers=cfg['num_layers'], output_size=cfg['output_size'], 
                    batch_first=cfg['batch_first'], bidirectional=cfg['bidirectional'])
    elif model_name == 'SingleGRU':
        return SingleGRU(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=128)
    elif model_name == 'HierGRU':
        return HierGRU(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'], 
                       num_layers=cfg['num_layers'], bidirectional=cfg['bidirectional'], 
                       batch_first=cfg['batch_first'])
    elif model_name == 'FCNetwork':
        return FCNetwork(input_size=len(cfg['applied_data_list'])*cfg['sequence_length']*cfg['in_feat_dim']
                         + cfg['gesture_type_embedding_dim'] + 1, 
                         hidden_size=cfg['hidden_size'], 
                         output_size=cfg['output_size'], 
                         use_batch_norm=cfg['use_batch_norm'],
                         applied_data_list=cfg['applied_data_list'])
    elif model_name == 'RNN':
        return
        
def create_recognition_model(model_name:str):
    """Use the model_name to create a model instance.
    Recognition model uses the touchscreen data to recognize the user's identity.
    Args:
        model_name (str): The name of the model.
    Raises:
        Exception: If the model_name is not in the model_config.yaml.
    Returns:
        model: The model instance.
    """
    model_config = load_model_config_yaml()
    if model_name in model_config.keys():
        cfg = model_config[model_name]
    else:
        raise Exception(f'No model named {model_name} in model_config.yaml')
    
    if model_name == 'FCNetwork':
        return FCNetwork(input_size=len(cfg['applied_data_list'])*cfg['sequence_length']*cfg['in_feat_dim']
                         + cfg['gesture_type_embedding_dim'] + 1, 
                         hidden_size=cfg['hidden_size'],
                         output_size=cfg['output_size'],
                         use_batch_norm=cfg['use_batch_norm'],
                         applied_data_list=cfg['applied_data_list'])
    elif model_name == 'SingleGRU':
        return SingleGRU(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'],
                         out_feat_dim=cfg['out_feat_dim'], num_layers=cfg['num_layers'],
                         use_all_outputs=cfg['use_all_outputs'],
                         applied_data_list=cfg['applied_data_list'], sequence_length=cfg['sequence_length'],
                         bidirectional=cfg['bidirectional'], batch_first=cfg['batch_first'])
        
    elif model_name == 'RNN':
        return RNN(model_type='recognition', input_size=cfg['input_size'], 
                   hidden_size=cfg['hidden_size'], output_size=cfg['output_size'], 
                   num_layers=cfg['num_layers'], bidirectional=cfg['bidirectional'],
                   batch_first=cfg['batch_first'], dropout=cfg['dropout'])
        
def create_test_model(model_name:str):
    test_config = load_test_config_yaml()
    if model_name in test_config.keys():
        cfg = test_config[model_name]
    else:
        raise Exception(f'No model named {model_name} in test_config.yaml')
    
    if model_name == 'FCNetwork':
        return FCNetwork(input_size=len(cfg['applied_data_list'])*cfg['sequence_length']*cfg['in_feat_dim']
                         + cfg['gesture_type_embedding_dim'] + 1, 
                         hidden_size=cfg['hidden_size'], 
                         output_size=cfg['output_size'], 
                         use_batch_norm=cfg['use_batch_norm'],
                         applied_data_list=cfg['applied_data_list'])
    elif model_name == 'SingleGRU':
        return SingleGRU(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'],
                         out_feat_dim=cfg['out_feat_dim'], num_layers=cfg['num_layers'],
                         use_all_outputs=cfg['use_all_outputs'],
                         applied_data_list=cfg['applied_data_list'], sequence_length=cfg['sequence_length'],
                         bidirectional=cfg['bidirectional'], batch_first=cfg['batch_first'])
        
    elif model_name == 'RNN':
        return RNN(input_size=cfg['input_size'], hidden_size=cfg['hidden_size'],
                   output_size=cfg['output_size'], rnn_type=cfg['rnn_type'],
                   num_layers=cfg['num_layers'], bidirectional=cfg['bidirectional'],
                   batch_first=cfg['batch_first'], dropout=cfg['dropout'])
        

def get_sensor_data_model(model_name:str):
    """Use the model_name to create a model instance.
    Sensor data model embeds the sensor data to a vector.
    Args:
        model_name (str): The name of the model.
    Raises:
        Exception: If the model_name is not in the model_config.yaml.
    Returns:
        model: The model instance.
    """
    model_config = load_model_config_yaml(cfg_file_name='sensor_model_config.yaml')
    if model_name in model_config.keys():
        cfg = model_config[model_name]
    else:
        raise Exception(f'No model named {model_name} in sensor_model_config.yaml')
    
    if model_name == 'FCNetwork':
        return FCNetwork(input_size=len(cfg['applied_data_list'])*cfg['sequence_length']*cfg['in_feat_dim']
                         + cfg['gesture_type_embedding_dim'] + 1, 
                         hidden_size=cfg['hidden_size'], 
                         output_size=cfg['output_size'], 
                         use_batch_norm=cfg['use_batch_norm'],
                         applied_data_list=cfg['applied_data_list'])
    elif model_name == 'SingleGRU':
        if cfg['use_all_outputs']:
            num_directions = 2 if cfg['bidirectional'] else 1
            out_dim = cfg['hidden_state_dim'] * num_directions * cfg['sequence_length']
        else:
            num_directions = 2 if cfg['bidirectional'] else 1
            out_dim = cfg['hidden_state_dim'] * num_directions
        
        return SingleGRU(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'],
                         num_layers=cfg['num_layers'], use_all_outputs=cfg['use_all_outputs'],
                         applied_data_list=cfg['applied_data_list'],
                         bidirectional=cfg['bidirectional'], data_type='sensor'), out_dim
    elif model_name == 'RNN':
        if cfg['use_all_outputs']:
            num_directions = 2 if cfg['bidirectional'] else 1
            out_dim = cfg['hidden_state_dim'] * num_directions * cfg['sequence_length']
        else:
            num_directions = 2 if cfg['bidirectional'] else 1
            out_dim = cfg['hidden_state_dim'] * num_directions
        
        return RNN(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'],
                   num_layers=cfg['num_layers'], use_all_outputs=cfg['use_all_outputs'],
                   applied_data_list=cfg['applied_data_list'],
                   bidirectional=cfg['bidirectional'], data_type='sensor'), out_dim



def get_touchscreen_data_model(model_name:str):
    """Use the model_name to create a model instance.
    Touchscreen data model embeds the touchscreen data to a vector.
    Args:
        model_name (str): The name of the model.
    Raises:
        Exception: If the model_name is not in the model_config.yaml.
    Returns:
        model: The model instance.
    """
    model_config = load_model_config_yaml(cfg_file_name='ts_model_config.yaml')
    if model_name in model_config.keys():
        cfg = model_config[model_name]
    else:
        raise Exception(f'No model named {model_name} in ts_model_config.yaml')
    
    if model_name == 'FCNetwork':
        return FCNetwork(input_size=len(cfg['applied_data_list'])*cfg['sequence_length']*cfg['in_feat_dim']
                         + cfg['gesture_type_embedding_dim'] + 1, 
                         hidden_size=cfg['hidden_size'], 
                         output_size=cfg['output_size'], 
                         use_batch_norm=cfg['use_batch_norm'],
                         applied_data_list=cfg['applied_data_list'])
    elif model_name == 'SingleGRU':
        if cfg['use_all_outputs']:
            num_directions = 2 if cfg['bidirectional'] else 1
            out_dim = cfg['hidden_state_dim'] * num_directions * cfg['sequence_length']
        else:
            num_directions = 2 if cfg['bidirectional'] else 1
            out_dim = cfg['hidden_state_dim'] * num_directions
        
        return SingleGRU(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'],
                         num_layers=cfg['num_layers'], use_all_outputs=cfg['use_all_outputs'],
                         applied_data_list=cfg['applied_data_list'],
                         bidirectional=cfg['bidirectional'], data_type='touchscreen'), out_dim
    elif model_name == 'RNN':
        if cfg['use_all_outputs']:
            num_directions = 2 if cfg['bidirectional'] else 1
            out_dim = cfg['hidden_state_dim'] * num_directions * cfg['sequence_length']
        else:
            num_directions = 2 if cfg['bidirectional'] else 1
            out_dim = cfg['hidden_state_dim'] * num_directions
        
        return RNN(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'],
                   num_layers=cfg['num_layers'], use_all_outputs=cfg['use_all_outputs'],
                   applied_data_list=cfg['applied_data_list'],
                   bidirectional=cfg['bidirectional'], data_type='touchscreen'), out_dim
