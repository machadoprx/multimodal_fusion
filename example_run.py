from src.models import multimodal_fusion_model

mf = multimodal_fusion_model('autoencoder', 5, 256, 128).get_model()
mf_2 = multimodal_fusion_model('add', 5, 256, 128).get_model()

print('Model 1')
mf.summary()

print('Model 2')
mf_2.summary()