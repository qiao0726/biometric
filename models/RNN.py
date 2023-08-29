import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, in_feat_dim=1, hidden_state_dim=32, num_layers=2, use_all_outputs=False,
                 bidirectional=False, applied_data_list=('hold_time', 'inter_time', 'distance', 'speed'),
                 data_type='sensor'):
        """ Simple RNN model for sensor and touch screen data.

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
            self.rnn = nn.RNN(input_size=len(self.apply_data_list)*in_feat_dim, hidden_size=hidden_state_dim,
                              num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        elif data_type == 'sensor':
            self.rnn = nn.RNN(input_size=in_feat_dim, hidden_size=hidden_state_dim,
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
        # output is a tuple, the first element is the output tensor, the second element is the hidden state
        # Shape of output[0] is (batch_size, sequence_length, hidden_size*num_directions)
        # Shape of output[1] is (num_layers*num_directions, batch_size, hidden_size)
        output = self.rnn(ts_data)
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
        output = self.rnn(sensor_data)
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