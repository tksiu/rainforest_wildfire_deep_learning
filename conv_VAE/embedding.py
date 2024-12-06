import tensorflow as tf
import numpy as np
import pickle
import gc


def get_CVAE_embeddings(model, layer_name, inputs, inputs_folder):

    cvae_vectors = []
    intermediate = tf.keras.Model(inputs = model.decoder.input, 
                                  outputs = model.decoder.get_layer(layer_name).output)
    
    for g in range(len(inputs)):

        with open(inputs_folder + inputs[g], "rb") as f:
            train_x = pickle.load(f)
        
        train_x = np.array([x for x in train_x])
        train_x = np.nan_to_num(train_x, nan=0)
        train_x = train_x / 255.0
        encoded_x = model.encoder.predict(train_x, batch_size=3)
        latent = intermediate.predict(encoded_x[2], batch_size=3)
        
        del train_x
        del encoded_x
        del latent
        # gc.collect()

        cvae_vectors.append(latent)
        return cvae_vectors
    
