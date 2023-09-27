# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'gesture_only' --use_gesture_type True --bidirectional False --out_feat_dim 16
# python -c "print('---------GRU-RNN-gesture_only-True-bidirectional-True-out_feat_dim-32---------')"
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'gesture_only' --use_gesture_type True --bidirectional False --out_feat_dim 32
# python -c "print('---------LSTM-GRU-gesture_only-True-bidirectional-False-out_feat_dim-64---------')"
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'gesture_only' --use_gesture_type True --bidirectional False --out_feat_dim 64
# python -c "print('---------RNN-LSTM-gesture_only-True-bidirectional-False-out_feat_dim-128---------')"

python train.py --sensor_model_name 'LSTM' --ts_model_name 'LSTM' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 64 --checkpoint_path '/home/qn/biometric/checkpoints/BioMetricNetwork_2023-09-27_05-03-38.pth'
python train.py --sensor_model_name 'LSTM' --ts_model_name 'LSTM' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 128 --checkpoint_path '/home/qn/biometric/checkpoints/BioMetricNetwork_2023-09-27_09-48-19.pth'
python train.py --sensor_model_name 'LSTM' --ts_model_name 'LSTM' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 256 --checkpoint_path '/home/qn/biometric/checkpoints/BioMetricNetwork_2023-09-27_11-26-48.pth'

# python train.py --sensor_model_name 'LSTM' --ts_model_name 'LSTM' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 512

# python train.py --sensor_model_name 'GRU' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 64
# python train.py --sensor_model_name 'GRU' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 128
# python train.py --sensor_model_name 'GRU' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 256
# python train.py --sensor_model_name 'GRU' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 512

# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 64
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 128
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 256
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 512

# #-----------------------------No bidirectional--------------------------------

# python train.py --sensor_model_name 'LSTM' --ts_model_name 'LSTM' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 64
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'LSTM' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 128
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'LSTM' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 256
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'LSTM' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 512

# python train.py --sensor_model_name 'GRU' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 64
# python train.py --sensor_model_name 'GRU' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 128
# python train.py --sensor_model_name 'GRU' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 256
# python train.py --sensor_model_name 'GRU' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 512

# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 64
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 128
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 256
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 512

# #-----------------------------More complicated sensor model--------------------------------


# python train.py --sensor_model_name 'GRU' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 64
# python train.py --sensor_model_name 'GRU' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 128
# python train.py --sensor_model_name 'GRU' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 256
# python train.py --sensor_model_name 'GRU' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 512


# python train.py --sensor_model_name 'LSTM' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 64
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 128
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 256
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 512


# python train.py --sensor_model_name 'LSTM' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 64
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 128
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 256
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 512

# #-----------------------------No bidirectional--------------------------------

# python train.py --sensor_model_name 'GRU' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 64
# python train.py --sensor_model_name 'GRU' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 128
# python train.py --sensor_model_name 'GRU' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 256
# python train.py --sensor_model_name 'GRU' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 512


# python train.py --sensor_model_name 'LSTM' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 64
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 128
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 256
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'GRU' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 512


# python train.py --sensor_model_name 'LSTM' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 64
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 128
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 256
# python train.py --sensor_model_name 'LSTM' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional False --out_feat_dim 512