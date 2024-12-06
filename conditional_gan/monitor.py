import tensorflow as tf
import numpy as np


class GAN_Monitor(tf.keras.callbacks.Callback):

    def __init__(self, gan_model, grid_features, grid_prev_targets):
        self.gan_model = gan_model
        self.grid_features = grid_features
        self.grid_prev_targets = grid_prev_targets
        
    def on_epoch_end(self, epoch, logs=False, log_path=None):

        out = []
        for i in range(0, len(self.grid_features), 2):
            random_latent_vectors = tf.random.normal([2, 6, 56, 56, 50])
            generated_images = self.gan_model.generator([random_latent_vectors, 
                                              self.grid_features[i:(i+2)].reshape(2, 6, 50, 50, 50*5), 
                                              self.grid_prev_targets[i:(i+2)].reshape(2, 112, 112, 1)], training=False)
            out.append(generated_images)
        out = np.concatenate(out)
        self.gan_model.predicted_imgs = out
        self.gan_model.fake_sim_embeddings = self.gan_model.cosine_distance()

        if logs:
            if (epoch+1) % 10 == 0:
                epoch_n = epoch+1
                self.gan_model.generator.save(
                    log_path + "epoch_{epoch_n}_generator.h5".format(epoch_n=epoch+1))
                self.gan_model.discriminator.save(
                    log_path + "epoch_{epoch_n}_discriminator.h5".format(epoch_n=epoch+1))
            
