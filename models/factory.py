from .LSTM import LSTM
from .FCNetwork import FCNetwork
from .GRU import GRU
from .RNN import RNN
from utils import load_model_config_yaml, load_test_config_yaml
# from models.networks import GestureClassificationNetwork, IDRecognitionNetwork, BioMetricNetwork, NoGestureNetwork



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
    elif model_name == 'GRU':
        return GRU(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=128)
    elif model_name == 'FCNetwork':
        return FCNetwork(input_size=len(cfg['applied_data_list'])*cfg['sequence_length']*cfg['in_feat_dim']
                         + cfg['gesture_type_embedding_dim'] + 1, 
                         hidden_size=cfg['hidden_size'], 
                         output_size=cfg['output_size'], 
                         use_batch_norm=cfg['use_batch_norm'],
                         applied_data_list=cfg['applied_data_list'])
    elif model_name == 'RNN':
        return
        
        

def get_sensor_data_model(model_name:str, bidirectional=False):
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
    
    if model_name == 'LSTM':
        if cfg['use_all_outputs']:
            num_directions = 2 if bidirectional else 1
            out_dim = cfg['hidden_state_dim'] * num_directions * cfg['sequence_length']
        else:
            num_directions = 2 if bidirectional else 1
            out_dim = cfg['hidden_state_dim'] * num_directions
        
        return LSTM(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'],
                   num_layers=cfg['num_layers'], use_all_outputs=cfg['use_all_outputs'],
                   applied_data_list=cfg['applied_data_list'],
                   bidirectional=bidirectional, data_type='sensor'), out_dim
    elif model_name == 'GRU':
        if cfg['use_all_outputs']:
            num_directions = 2 if bidirectional else 1
            out_dim = cfg['hidden_state_dim'] * num_directions * cfg['sequence_length']
        else:
            num_directions = 2 if bidirectional else 1
            out_dim = cfg['hidden_state_dim'] * num_directions
        
        return GRU(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'],
                         num_layers=cfg['num_layers'], use_all_outputs=cfg['use_all_outputs'],
                         applied_data_list=cfg['applied_data_list'],
                         bidirectional=bidirectional, data_type='sensor'), out_dim
    elif model_name == 'RNN':
        if cfg['use_all_outputs']:
            num_directions = 2 if bidirectional else 1
            out_dim = cfg['hidden_state_dim'] * num_directions * cfg['sequence_length']
        else:
            num_directions = 2 if bidirectional else 1
            out_dim = cfg['hidden_state_dim'] * num_directions
        
        return RNN(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'],
                   num_layers=cfg['num_layers'], use_all_outputs=cfg['use_all_outputs'],
                   applied_data_list=cfg['applied_data_list'],
                   bidirectional=bidirectional, data_type='sensor'), out_dim



def get_touchscreen_data_model(model_name:str, bidirectional=False):
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
    
    if model_name == 'LSTM':
        if cfg['use_all_outputs']:
            num_directions = 2 if bidirectional else 1
            out_dim = cfg['hidden_state_dim'] * num_directions * cfg['sequence_length']
        else:
            num_directions = 2 if bidirectional else 1
            out_dim = cfg['hidden_state_dim'] * num_directions
        
        return LSTM(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'],
                   num_layers=cfg['num_layers'], use_all_outputs=cfg['use_all_outputs'],
                   applied_data_list=cfg['applied_data_list'],
                   bidirectional=bidirectional, data_type='touchscreen'), out_dim
    
    elif model_name == 'GRU':
        if cfg['use_all_outputs']:
            num_directions = 2 if bidirectional else 1
            out_dim = cfg['hidden_state_dim'] * num_directions * cfg['sequence_length']
        else:
            num_directions = 2 if bidirectional else 1
            out_dim = cfg['hidden_state_dim'] * num_directions
        
        return GRU(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'],
                         num_layers=cfg['num_layers'], use_all_outputs=cfg['use_all_outputs'],
                         applied_data_list=cfg['applied_data_list'],
                         bidirectional=bidirectional, data_type='touchscreen'), out_dim
    
    elif model_name == 'RNN':
        if cfg['use_all_outputs']:
            num_directions = 2 if bidirectional else 1
            out_dim = cfg['hidden_state_dim'] * num_directions * cfg['sequence_length']
        else:
            num_directions = 2 if bidirectional else 1
            out_dim = cfg['hidden_state_dim'] * num_directions
        
        return RNN(in_feat_dim=cfg['in_feat_dim'], hidden_state_dim=cfg['hidden_state_dim'],
                   num_layers=cfg['num_layers'], use_all_outputs=cfg['use_all_outputs'],
                   applied_data_list=cfg['applied_data_list'],
                   bidirectional=bidirectional, data_type='touchscreen'), out_dim


    