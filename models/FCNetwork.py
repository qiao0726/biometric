import torch
import torch.nn as nn

# Define your feature extractor model
class FCNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, 
                 output_size, use_batch_norm=True, gesture_type_embedding_dim=8,
                 applied_data_list=('hold_time', 'inter_time', 'distance', 'speed'),
                 data_type='touchscreen'):
        super(FCNetwork, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.gesture_type_embedding = nn.Embedding(14, gesture_type_embedding_dim)
        
        self.use_batch_norm = use_batch_norm
        self.apply_data_list = applied_data_list
        self.data_type = data_type

    def forward(self, sensor_data, ts_data, total_time, gesture_type):
        if self.data_type == 'sensor':
            return self.forward_sensor(sensor_data)
        elif self.data_type == 'touchscreen':
            return self.forward_ts(ts_data)
    
    def forward_ts(self, ts_data):
        hold_time, inter_time, distance, speed = ts_data
        # Reshape to [batch_size, sequence_length, 1]
        hold_time.unsqueeze_(-1)
        inter_time.unsqueeze_(-1)
        distance.unsqueeze_(-1)
        speed.unsqueeze_(-1)
        #gesture_type.unsqueeze_(-1)
        
        applied_data_list = list([var for var_name, var in locals().items() if var_name in self.apply_data_list])
        # Embedding the gesture type code
        # gesture_type_embedding = self.gesture_type_embedding(gesture_type.long())
        
        #applied_data_list.append(gesture_type_embedding)
        concat_tensor = torch.cat(applied_data_list, dim=-1)
        flatten_tensor = concat_tensor.reshape(concat_tensor.shape[0], -1)
        
        concat_all_tensor = torch.cat((flatten_tensor, total_time, gesture_type_embedding), dim=-1)
        
        output = self.fc(concat_all_tensor)
        output = self.bn(output) if self.use_batch_norm else output
        output = self.relu(output)
        output = self.output(output)
        return output
    
    def forward_sensor(self, sensor_data):
        output = self.fc(sensor_data)
        output = self.bn(output) if self.use_batch_norm else output
        output = self.relu(output)
        output = self.output(output)
        return output
    
    