import tensorflow as tf
from tensorflow.keras.layers import Dot,Lambda,Input, Activation, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot,Activation,Dense, Input, concatenate, multiply, average, subtract, add, Dropout, Lambda
from tensorflow.keras.models import Model

from .metrics import f1_m

class multimodal_fusion_model:
    operators = [
        'autoencoder',
        'add',
        'subtract',
        'average',
        'multiply',
        'concatenate'
        'att',
        'att_labels',
    ]
    def __init__(self, operator, num_classes, emb_size_1, emb_size_2):
        if operator not in self.operators:
            raise('Operator not implemented')
        self.operator = operator
        self.num_classes = num_classes
        self.emb_size_1 = emb_size_1
        self.emb_size_2 = emb_size_2
    
    def get_model(self):
        if self.operator == 'autoencoder':
            return multimodal_fusion_model.autoencoder_att_labels(self.num_classes, self.emb_size_2, self.emb_size_1)
        else: # size_1, size_2,num_classes,operator='concatenate'
            return multimodal_fusion_model.multimodal_text_image(self.emb_size_1, self.emb_size_2, self.num_classes, operator=self.operator)

    @staticmethod
    def autoencoder_att_labels(num_classes, size_1, size_2):
        input_img = Input(shape=(size_1,))
        input_txt = Input(shape=(size_2,))
        fusion_dim = 512
    
        im_emb = Activation('tanh')(input_img)
        im_emb = Dense(fusion_dim, activation='tanh')(im_emb)
    
        txt_emb = Activation('tanh')(input_txt)
        txt_emb = Dense(fusion_dim, activation='tanh')(txt_emb)
    
        ''' Attention Modality '''
        #[input_1, input_2] = [visual_embd, average_seq]
        input_1 = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))(im_emb) # (bs, ndim)
        input_2 = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))(txt_emb) # (bs, ndim)
        

        output_size=1
        # Step 1. Get scalar weights
        scalar_input_1 = Dense(output_size)(input_1)  # (batch_size, output_size)
        scalar_input_2 = Dense(output_size)(input_2)  # (batch_size, output_size)
        scalar_input_1_exp = Lambda(lambda x: tf.keras.backend.expand_dims(x))(scalar_input_1)  # (batch_size, output_size, 1)
        scalar_input_2_exp = Lambda(lambda x: tf.keras.backend.expand_dims(x))(scalar_input_2)  # (batch_size, output_size, 1)
        scalars = concatenate([scalar_input_1_exp, scalar_input_2_exp], name='concat')  # (batch_size, output_size, 2)
        
        # # Step 2. Normalize weights - softmax
        alphas = Activation('softmax')(scalars)  # (batch_size, output_size, 2)
        
        # Step 3. Weighted average
        input_1_exp = Lambda(lambda x: tf.keras.backend.expand_dims(x))(input_1)  # (batch_size, nb_feats, 1)
        input_2_exp = Lambda(lambda x: tf.keras.backend.expand_dims(x))(input_2)  # (batch_size, nb_feats, 1)
        features = concatenate([input_1_exp, input_2_exp], name='concat_feats')  # (batch_size, nb_feats, 2)
    
        latent = Dot(axes=[-1, -1])([alphas, features])  # (batch_size, output_size, nb_feats)
        latent = tf.reduce_mean(latent, axis=1)
        
        encoder = Model([input_txt, input_img], [latent,alphas], name='encoder')
        
        clf_in = Input(shape=(fusion_dim,))
        clf_probs = Dropout(0.4)(clf_in)
        clf_probs = Dense(num_classes, activation='softmax')(clf_probs) # (batch_size, nb_labels)
        clf = Model(clf_in, clf_probs, name='clf')
    
        decoder_in = Input(shape=(fusion_dim,))
        im_rebuild = Dense(fusion_dim, activation='tanh')(decoder_in)
        im_rebuild = Dense(size_1, name='img_reb')(im_rebuild)
    
        txt_rebuild = Dense(fusion_dim, activation='tanh')(decoder_in)
        txt_rebuild = Dense(size_2, name='txt_reb')(txt_rebuild)
    
        decoder = Model(decoder_in, outputs=[txt_rebuild,im_rebuild], name='decoder')
    
        autoencoder = Model([input_txt, input_img], [decoder(encoder([input_txt, input_img])[0]), clf(encoder([input_txt, input_img])[0]) ] )

        autoencoder.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=5e-4),
                        loss=['mse','mse','categorical_crossentropy'],
                        metrics=['accuracy'],
                        loss_weights=[2.0,2.0,0.1]) # max_losses(512,512,~15) decoder_loss: 0.0021 - decoder_1_loss: 0.8902 - clf_loss: 1.8469

        return autoencoder

    @staticmethod
    def multimodal_text_image(size_1, size_2,num_classes,operator='concatenate',verbose=0):

        # fusion_dim = X_1.shape[1]+X_2.shape[1]
        fusion_dim = size_1

        inp1 = Input(shape=(size_1))
        inp2 = Input(shape=(size_2))

        l1 = Dense(fusion_dim, activation='relu')(inp1)
        l2 = Dense(fusion_dim, activation='relu')(inp2)
        # l1 = inp1
        # l2 = inp2

        # fusion layer
        if(operator=='concatenate'):
            w = concatenate([l1,l2])
        if(operator=='multiply'):
            w = multiply([l1,l2])
        if(operator=='average'):
            w = average([l1,l2])
        if(operator=='subtract'):
            w = subtract([l1,l2])
        if(operator=='add'):
            w = add([l1,l2])
        if(operator=='att'):
            visual_embd = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))(l1) # (bs, ndim)
            average_seq = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))(l2) # (bs, ndim)
            scalar_visual = Dense(1)(visual_embd) # (bs, 1)
            scalar_text = Dense(1)(average_seq) # (bs, 1)
            scalars = concatenate([scalar_visual, scalar_text], name='concat')  # (bs, 2)

            # # Step 2. Normalize weights - softmax
            alphas = Activation('softmax')(scalars) # (bs, 2)

            # Step 3. Weighted average
            visual_embd_2 = Lambda( lambda x: tf.keras.backend.expand_dims(x) ) (visual_embd) # (bs, ndim, 1)
            average_seq_2 = Lambda( lambda x: tf.keras.backend.expand_dims(x) )(average_seq) # (bs, ndim, 1)
            features = concatenate([visual_embd_2, average_seq_2], name='concat_feats') # (bs, ndim, 2)
            w = Dot(axes=[-1, -1])([alphas, features]) # (bs, ndim)
        if(operator=='att_labels'):
            #[input_1, input_2] = [visual_embd, average_seq]
            input_1 = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))(l1) # (bs, ndim)
            input_2 = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))(l2) # (bs, ndim)
            output_size=num_classes
            # Step 1. Get scalar weights
            scalar_input_1 = Dense(output_size)(input_1)  # (batch_size, nb_labels)
            scalar_input_2 = Dense(output_size)(input_2)  # (batch_size, nb_labels)
            scalar_input_1_exp = Lambda(lambda x: tf.keras.backend.expand_dims(x))(scalar_input_1)  # (batch_size, nb_labels, 1)
            scalar_input_2_exp = Lambda(lambda x: tf.keras.backend.expand_dims(x))(scalar_input_2)  # (batch_size, nb_labels, 1)
            scalars = concatenate([scalar_input_1_exp, scalar_input_2_exp], name='concat')  # (batch_size, nb_labels, 2)
            
            # # Step 2. Normalize weights - softmax
            alphas = Activation('softmax')(scalars)  # (batch_size, nb_labels, 2)
            
            # Step 3. Weighted average
            input_1_exp = Lambda(lambda x: tf.keras.backend.expand_dims(x))(input_1)  # (batch_size, nb_feats, 1)
            input_2_exp = Lambda(lambda x: tf.keras.backend.expand_dims(x))(input_2)  # (batch_size, nb_feats, 1)
            features = concatenate([input_1_exp, input_2_exp], name='concat_feats')  # (batch_size, nb_feats, 2)
            w = Dot(axes=[-1, -1])([alphas, features])  # (batch_size, nb_labels, nb_feats)

        w = Dropout(0.5)(w)
        # fusion_layer = Dense(fusion_dim, activation='relu')(w)
        fusion_layer = w

        if (operator == 'att_labels'): # nm: new
            output = Dense(1)(fusion_layer)  # (batch_size, nb_labels, 1)  
            output = Lambda(lambda x: tf.keras.backend.squeeze(x, axis=-1))(output)  # (batch_size, nb_labels)
            output = Activation('softmax')(output)  # (batch_size, nb_labels)    
        else:
            output = Dense(num_classes,activation='softmax')(fusion_layer)

        model = Model(inputs=[inp1, inp2], outputs=output)
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m])

        return model

