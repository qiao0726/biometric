import torch.nn as nn
import torch


class HierGRU(nn.Module):
    def __init__(self, in_feat_dim, hidden_state_dim, out_feat_dim,
                 num_layers=2, bidirectional=False, batch_first=True,
                 applied_data_list=('hold_time', 'inter_time', 'distance', 'speed')):
        """ Hierarchical GRU model for sensor and touch screen data.

        Args:
            in_feat_dim (int): the dimensionality of the input features at each time step.
            hidden_state_dim (int): the dimensionality of the hidden states and output features.
            num_layers (int, optional): number of layers. Defaults to 2.
            bidirectional (bool, optional): if use bi-GRU. Defaults to False.
            batch_first (bool, optional): Defaults to True.
            applied_data_list (tuple, optional): Determine which data to be applied. Defaults to ('hold_time', 'inter_time', 'pressure', 'distance', 'speed').
        """
        super().__init__()
        # input of the main_gru is the concatenation of the outputs of all sub_grus
        self.main_gru = nn.GRU(input_size=hidden_state_dim*len(applied_data_list), hidden_size=hidden_state_dim, 
                          num_layers=num_layers, bidirectional=bidirectional, batch_first=batch_first)
        
        self.sub_grus = dict()
        for data_name in applied_data_list:
            self.sub_grus[data_name] = nn.GRU(input_size=in_feat_dim, hidden_size=hidden_state_dim,
                                              num_layers=num_layers, bidirectional=bidirectional, 
                                              batch_first=batch_first, return_sequences=True)
            
        self.fc = nn.Linear(hidden_state_dim, out_feat_dim)
        
    def forward(self, sensor_data_dict, touchscreen_data_dict):
        sub_grus_output = list()
        for data_name in self.sub_grus.keys():
            # Get the output of this sensor data
            # The output tensor will have shape (batch_size, sequence_length, hidden_size*num_directions)
            this_gru_output, this_gru_hidden = self.sub_grus[data_name](sensor_data_dict[data_name])
            sub_grus_output.append(this_gru_output)
        
        main_gru_input = torch.cat(sub_grus_output, dim=2)
        return


class SingleGRU(nn.Module):
    def __init__(self, in_feat_dim=1, hidden_state_dim=32, num_layers=2, use_all_outputs=False,
                 bidirectional=False, applied_data_list=('hold_time', 'inter_time', 'distance', 'speed'),
                 data_type='sensor'):
        """ Single GRU model for sensor and touch screen data.

        Args:
            in_feat_dim (int): the dimensionality of the input features at each time step.
            hidden_state_dim (int): the dimensionality of the hidden states and output features.
            num_layers (int, optional): number of layers. Defaults to 2.
            bidirectional (bool, optional): if use bi-GRU. Defaults to False.
            batch_first (bool, optional): Defaults to True.
            use_all_outputs (bool, optional): if use all the outputs of the GRU. Defaults to False.
            applied_data_list (tuple, optional): Determine which data to be applied. Defaults to ('hold_time', 'inter_time', 'distance', 'speed').
            data_type (str, optional): 'sensor' or 'touchscreen'. Defaults to 'sensor'.
        """
        super().__init__()
        self.apply_data_list = applied_data_list
        if data_type == 'touchscreen':
            self.main_gru = nn.GRU(input_size=len(self.apply_data_list)*in_feat_dim, hidden_size=hidden_state_dim, 
                                num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        elif data_type == 'sensor':
            self.main_gru = nn.GRU(input_size=in_feat_dim, hidden_size=hidden_state_dim,
                                   num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        else:
            raise Exception('data_type must be one of "sensor" and "touchscreen"')
        self.use_all_outputs = use_all_outputs
        self.data_type = data_type
        
    def forward(self, ts_data, sensor_data):
        if self.data_type == 'sensor':
            return self.forward_sensor(sensor_data)
        elif self.data_type == 'touchscreen':
            return self.forward_ts(ts_data)
        else:
            raise Exception('data_type must be one of "sensor" and "touchscreen"')
    
    def forward_ts(self, ts_data):
        hold_time, inter_time, distance, speed = ts_data
        
        # Reshape to [batch_size, sequence_length, 1]
        hold_time.unsqueeze_(-1)
        inter_time.unsqueeze_(-1)
        distance.unsqueeze_(-1)
        speed.unsqueeze_(-1)
        
        applied_data_list = list([var for var_name, var in locals().items() if var_name in self.apply_data_list])
        
        concat_tensor = torch.cat(applied_data_list, dim=-1)
        
        # output is a tuple, the first element is the output tensor, the second element is the hidden state
        # Shape of output[0] is (batch_size, sequence_length, hidden_size*num_directions)
        # Shape of output[1] is (num_layers*num_directions, batch_size, hidden_size)
        output = self.main_gru(concat_tensor)
        output = output[0]
        # --------------------Use all the outputs of the GRU---------------------
        if self.use_all_outputs:
            # reshape output to [batch_size, hidden_size*num_directions*sequence_length]
            output = output.reshape(output.shape[0], -1)
        # -----------------Use the last output of the GRU----------------------------
        else:
            # shape is (batch_size, hidden_size*num_directions*1)
            output = output[:, -1, :]
        
        return output
    
    def forward_sensor(self, sensor_data):
        # output is a tuple, the first element is the output tensor, the second element is the hidden state
        # Shape of output[0] is (batch_size, sequence_length, hidden_size*num_directions)
        # Shape of output[1] is (num_layers*num_directions, batch_size, hidden_size)
        output = self.main_gru(sensor_data)
        output = output[0]
        # --------------------Use all the outputs of the GRU---------------------
        if self.use_all_outputs:
            # reshape output to [batch_size, hidden_size*num_directions*sequence_length]
            output = output.reshape(output.shape[0], -1)
        # -----------------Use the last output of the GRU----------------------------
        else:
            # shape is (batch_size, hidden_size*num_directions*1)
            output = output[:, -1, :]
        return output