from src.models import multimodal_fusion_model
import numpy as np
import tensorflow as tf

dataset = 'youtube'

X1_train = np.load(f'./data/{dataset}/X1_train.npy')
X1_val = np.load(f'./data/{dataset}/X1_val.npy')
X1_test = np.load(f'./data/{dataset}/X1_test.npy')
X2_train = np.load(f'./data/{dataset}/X2_train.npy')
X2_val = np.load(f'./data/{dataset}/X2_val.npy')
X2_test = np.load(f'./data/{dataset}/X2_test.npy')
y_train = np.load(f'./data/{dataset}/y_train.npy')
y_val = np.load(f'./data/{dataset}/y_val.npy')
y_test = np.load(f'./data/{dataset}/y_test.npy')

mf = multimodal_fusion_model('add', y_train.shape[1], X1_train.shape[1], X2_train.shape[1]).get_model()

mf.summary()
mf.fit([X1_train,X2_train], y_train, validation_data=([X1_val, X2_val], y_val),
                epochs=1000,
                batch_size=16,
                shuffle=True)

# Autoencoder
#mf.fit([X1_train, X2_train], [[X1_train, X2_train], y_train], epochs=1000, batch_size=64)