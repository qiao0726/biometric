# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'gesture_only' --use_gesture_type True --bidirectional False --out_feat_dim 16
# python -c "print('---------GRU-RNN-gesture_only-True-bidirectional-True-out_feat_dim-32---------')"
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'gesture_only' --use_gesture_type True --bidirectional False --out_feat_dim 32
# python -c "print('---------LSTM-GRU-gesture_only-True-bidirectional-False-out_feat_dim-64---------')"
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'gesture_only' --use_gesture_type True --bidirectional False --out_feat_dim 64
# python -c "print('---------RNN-LSTM-gesture_only-True-bidirectional-False-out_feat_dim-128---------')"

# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'id_only' --use_gesture_type True --bidirectional False --out_feat_dim 32
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'id_only' --use_gesture_type True --bidirectional False --out_feat_dim 64
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'id_only' --use_gesture_type True --bidirectional False --out_feat_dim 128

python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'no_gesture' --use_gesture_type False --bidirectional False --out_feat_dim 16
python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'no_gesture' --use_gesture_type False --bidirectional False --out_feat_dim 32
python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'no_gesture' --use_gesture_type False --bidirectional False --out_feat_dim 64
python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'no_gesture' --use_gesture_type False --bidirectional False --out_feat_dim 128

# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'no_gesture' --use_gesture_type False --bidirectional True --out_feat_dim 16
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'no_gesture' --use_gesture_type False --bidirectional True --out_feat_dim 32
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'no_gesture' --use_gesture_type False --bidirectional True --out_feat_dim 64
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'no_gesture' --use_gesture_type False --bidirectional True --out_feat_dim 128

# python train.py --sensor_model_name 'SingleGRU' --ts_model_name 'SingleGRU' --model_type 'no_gesture' --use_gesture_type False --bidirectional False --out_feat_dim 16
# python train.py --sensor_model_name 'SingleGRU' --ts_model_name 'SingleGRU' --model_type 'no_gesture' --use_gesture_type False --bidirectional False --out_feat_dim 32
# python train.py --sensor_model_name 'SingleGRU' --ts_model_name 'SingleGRU' --model_type 'no_gesture' --use_gesture_type False --bidirectional False --out_feat_dim 64
# python train.py --sensor_model_name 'SingleGRU' --ts_model_name 'SingleGRU' --model_type 'no_gesture' --use_gesture_type False --bidirectional False --out_feat_dim 128

# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'id_only' --use_gesture_type False --bidirectional False --out_feat_dim 16
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'id_only' --use_gesture_type False --bidirectional False --out_feat_dim 32
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'id_only' --use_gesture_type False --bidirectional False --out_feat_dim 64
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'id_only' --use_gesture_type False --bidirectional False --out_feat_dim 128

# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 16
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 32
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 64
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'both' --use_gesture_type True --bidirectional True --out_feat_dim 128

# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'id_only' --use_gesture_type False --bidirectional True --out_feat_dim 16
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'id_only' --use_gesture_type False --bidirectional True --out_feat_dim 32
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'id_only' --use_gesture_type False --bidirectional True --out_feat_dim 64
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'id_only' --use_gesture_type False --bidirectional True --out_feat_dim 128


# python train.py --sensor_model_name 'LSTM' --ts_model_name 'GRU' --model_type 'gesture_only' --use_gesture_type True --bidirectional True --out_feat_dim 32
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'gesture_only' --use_gesture_type True --bidirectional True --out_feat_dim 64
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'gesture_only' --use_gesture_type True --bidirectional True --out_feat_dim 128

# python train.py --sensor_model_name 'LSTM' --ts_model_name 'GRU' --model_type 'gesture_only' --use_gesture_type True --bidirectional True --out_feat_dim 32
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'gesture_only' --use_gesture_type True --bidirectional True --out_feat_dim 64
# python train.py --sensor_model_name 'RNN' --ts_model_name 'RNN' --model_type 'gesture_only' --use_gesture_type True --bidirectional True --out_feat_dim 128