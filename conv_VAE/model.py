import tensorflow as tf


class ConvVAE_1120(tf.keras.Model):
    """Convolutional variational autoencoder"""
    """for burned area (long term), ndvi & change images, land cover & change images"""

    def __init__(self, width, height, depth, reduced_width, reduced_height, latent_dim):
        super(ConvVAE_1120, self).__init__()
        self.width = int(width)
        self.height = int(height)
        self.reduced_width = int(reduced_width)
        self.reduced_height = int(reduced_height)
        self.depth = int(depth)
        self.latent_dim = int(latent_dim)

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    class Sampling(tf.keras.layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_encoder(self):
        encoder_inputs = tf.keras.Input(shape=(self.width, self.height, self.depth))
        x = tf.keras.layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.ZeroPadding2D(padding=(5,5))(x)
        x = tf.keras.layers.Conv2D(16, 2, activation="relu", strides=3, padding="same")(x)
        x = tf.keras.layers.Conv2D(8, 2, activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.Conv2D(4, 2, activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.latent_dim, activation="relu")(x)
        z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = self.Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder
    
    def get_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense((self.reduced_width // 2) * (self.reduced_height // 2) * 8, activation="relu")(latent_inputs)
        x = tf.keras.layers.Reshape((self.reduced_width // 2, self.reduced_height // 2, 8))(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=1, padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.ZeroPadding2D(padding=(3,3))(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=5, padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(8, 3, strides=2, padding="same")(x)
        decoder_outputs = tf.keras.layers.Conv2DTranspose(self.depth, 1, strides=1, activation="sigmoid", padding="same")(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return decoder

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            reconstruction_loss = reconstruction_loss / (data.shape[0] * data.shape[1])
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

