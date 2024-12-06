import tensorflow as tf
import numpy as np
from keras import backend as K
from sklearn.metrics import accuracy_score, jaccard_score
from sklearn.metrics.pairwise import cosine_similarity


####  Generator:   Res-BiConvLSTM Encoder and Noises
class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()
        self.generator = self.generator_model()

    def generator_model(self):
        """ Input 1) BiConvLSTM encoded embeddings from lagged land-grid weather/environmental features; 
                  2) Gaussian random noises;  
                  3) Previous time-step wildfire image;
        """
        input_feat = tf.keras.layers.Input(shape=(6, 50, 50, 50*5))
        input_gen = tf.keras.layers.Input(shape=(6, 56, 56, 50))
        input_wf = tf.keras.layers.Input(shape=(112, 112, 1))

        """  Stacked Res-BiConvLSTM : for known features
        """
        feat = tf.keras.layers.ZeroPadding3D((0,3,3))(input_feat)

        convlstm_1 = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(1,1), strides=(1,1), 
                                                padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
        bi_convlstm_1 = tf.keras.layers.Bidirectional(convlstm_1)
        x1 = bi_convlstm_1(feat)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        
        convlstm_2a = tf.keras.layers.ConvLSTM2D(filters=256, kernel_size=(3,3), strides=(2,2), 
                                                 padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
        bi_convlstm_2a = tf.keras.layers.Bidirectional(convlstm_2a)
        convlstm_2b = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(1,1), strides=(1,1), 
                                                 padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
        bi_convlstm_2b = tf.keras.layers.Bidirectional(convlstm_2b)
        x2 = bi_convlstm_2b(bi_convlstm_2a(x1))
        x1_convert = tf.keras.layers.MaxPool3D((1,2,2))(x1)
        x2 = tf.math.add(x1_convert, x2)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        x2 = tf.keras.layers.LeakyReLU()(x2)

        convlstm_3a = tf.keras.layers.ConvLSTM2D(filters=256, kernel_size=(3,3), strides=(2,2), 
                                                 padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
        bi_convlstm_3a = tf.keras.layers.Bidirectional(convlstm_3a)
        convlstm_3b = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(1,1), strides=(1,1), 
                                                 padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
        bi_convlstm_3b = tf.keras.layers.Bidirectional(convlstm_3b)
        x3 = bi_convlstm_3b(bi_convlstm_3a(x2))
        x2_convert = tf.keras.layers.MaxPool3D((1,2,2))(x2)
        x3 = tf.math.add(x2_convert, x3)
        x3 = tf.keras.layers.BatchNormalization()(x3)
        x3 = tf.keras.layers.LeakyReLU()(x3)

        """  REPEAT for encoding random noises
        """
        convlstm_1 = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3,3), strides=(2,2), 
                                                padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
        bi_convlstm_1 = tf.keras.layers.Bidirectional(convlstm_1)
        z1 = bi_convlstm_1(input_gen)
        z1 = tf.keras.layers.BatchNormalization()(z1)
        z1 = tf.keras.layers.LeakyReLU()(z1)

        convlstm_2 = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3,3), strides=(2,2), 
                                                padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
        bi_convlstm_2 = tf.keras.layers.Bidirectional(convlstm_2)
        z2 = bi_convlstm_2(z1)
        z2 = tf.keras.layers.BatchNormalization()(z2)
        z2 = tf.keras.layers.LeakyReLU()(z2)

        """  3D convolutions
        """
        g = tf.keras.layers.Concatenate()([x3, z2])
        g = tf.keras.layers.Conv3D(filters=768, kernel_size=(3,1,1), strides=(2,1,1), padding="same")(g)
        g = tf.keras.layers.BatchNormalization()(g)
        g = tf.keras.layers.LeakyReLU()(g)
        g = tf.keras.layers.Conv3D(filters=1536, kernel_size=(3,1,1), strides=(3,1,1), padding="same")(g)
        g = tf.keras.layers.BatchNormalization()(g)
        g = tf.keras.layers.LeakyReLU()(g)

        g = tf.keras.layers.Reshape(target_shape=(14, 14, 1536))(g)
        g = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=(1,1), padding="same")(g)

        """  Deconvolutions
        """
        deconv_0 = tf.keras.layers.Conv2DTranspose(512, kernel_size=(3,3), strides=(2,2), 
                                                   padding='same', use_bias=False)(g)
        deconv_0 = tf.keras.layers.BatchNormalization()(deconv_0)
        deconv_0 = tf.keras.layers.LeakyReLU()(deconv_0)
        
        deconv_1 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3,3), strides=(2,2), 
                                                   padding='same', use_bias=False)(deconv_0)
        deconv_1 = tf.keras.layers.BatchNormalization()(deconv_1)
        deconv_1 = tf.keras.layers.LeakyReLU()(deconv_1)

        deconv_2 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3,3), strides=(2,2), 
                                                   padding='same', use_bias=False)(deconv_1)
        deconv_2 = tf.keras.layers.BatchNormalization()(deconv_2)
        deconv_2 = tf.keras.layers.LeakyReLU()(deconv_2)

        condition_wf = tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), 
                                              padding='same', use_bias=False)(input_wf)
        deconv_2 = tf.math.add(deconv_2, condition_wf)

        deconv_3 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3,3), strides=(1,1), 
                                                   padding='same', use_bias=False)(deconv_2)
        deconv_3 = tf.keras.layers.BatchNormalization()(deconv_3)
        deconv_3 = tf.keras.layers.LeakyReLU()(deconv_3)
        
        deconv_4 = tf.keras.layers.Conv2DTranspose(16, kernel_size=(1,1), strides=(1,1), 
                                                   padding='same', use_bias=False)(deconv_3)
        deconv_4 = tf.keras.layers.BatchNormalization()(deconv_4)
        deconv_4 = tf.keras.layers.LeakyReLU()(deconv_4)
        
        """  Outputs
        """
        target = tf.keras.layers.Conv2DTranspose(1, kernel_size=(1,1), strides=(1,1), 
                                                 padding='same', use_bias=False, activation='sigmoid')(deconv_4)
        
        model = tf.keras.Model(inputs = [input_gen, input_feat, input_wf], outputs = target)

        return model




####  Discriminator:  Conditioning on previous lagged features
class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = self.discriminator_model()

    def discriminator_model(self):
        """ Input 1) real/fake wildfire images at time T;  
                  2) conditional information from known features at time T - 6 to T - 1
        """
        input_wf = tf.keras.layers.Input(shape=(112, 112, 1))
        input_prev_wf = tf.keras.layers.Input(shape=(112, 112, 1))
        input_similarity = tf.keras.layers.Input(shape=(270, 270, 1))
        
        x_wf = tf.keras.layers.Concatenate()([input_wf, input_prev_wf])
        x_wf = tf.keras.layers.Conv2D(16, kernel_size=(3,3), strides=2, padding='same')(x_wf)

        x_wf1 = tf.keras.layers.Conv2D(16, kernel_size=(1,3), strides=1, padding='same')(x_wf)
        x_wf2 = tf.keras.layers.Conv2D(16, kernel_size=(3,1), strides=1, padding='same')(x_wf)
        x_wf3 = tf.keras.layers.Conv2D(16, kernel_size=(3,3), strides=1, padding='same')(x_wf)
        x_wf4 = tf.keras.layers.Conv2D(48, kernel_size=(3,3), strides=1, padding='same')(x_wf)
        x_wf = tf.keras.layers.Concatenate()([x_wf1, x_wf2, x_wf3])
        x_wf = tf.math.add(x_wf, x_wf4)
        x_wf = tf.keras.layers.BatchNormalization()(x_wf)
        x_wf = tf.keras.layers.LeakyReLU()(x_wf)

        x_wf = tf.keras.layers.Conv2D(48, kernel_size=(3,3), strides=2, padding='same')(x_wf)

        x_wf1 = tf.keras.layers.Conv2D(48, kernel_size=(1,3), strides=1, padding='same')(x_wf)
        x_wf2 = tf.keras.layers.Conv2D(48, kernel_size=(3,1), strides=1, padding='same')(x_wf)
        x_wf3 = tf.keras.layers.Conv2D(48, kernel_size=(3,3), strides=1, padding='same')(x_wf)
        x_wf4 = tf.keras.layers.Conv2D(144, kernel_size=(3,3), strides=1, padding='same')(x_wf)
        x_wf = tf.keras.layers.Concatenate()([x_wf1, x_wf2, x_wf3])
        x_wf = tf.math.add(x_wf, x_wf4)
        x_wf = tf.keras.layers.BatchNormalization()(x_wf)
        x_wf = tf.keras.layers.LeakyReLU()(x_wf)

        x_wf = tf.keras.layers.Conv2D(288, kernel_size=(3,3), strides=2, padding='same')(x_wf)
        
        x_wf = tf.keras.layers.Dense(128)(x_wf)
        x_wf = tf.keras.layers.Dense(64)(x_wf)
        x_wf = tf.keras.layers.Dense(32)(x_wf)
        x_wf = tf.keras.layers.Dense(24)(x_wf)
        x_wf = tf.keras.layers.BatchNormalization()(x_wf)
        x_wf = tf.keras.layers.LeakyReLU()(x_wf)
        x_wf = tf.keras.layers.Dropout(0.1)(x_wf)

        diff = tf.keras.layers.Conv2D(5, kernel_size=(5,5), strides=5, padding='same')(input_similarity)
        diff = tf.keras.layers.Conv2D(10, kernel_size=(5,5), strides=2, padding='same')(diff)
        diff = tf.keras.layers.Conv2D(20, kernel_size=(5,5), strides=2, padding='same')(diff)

        x_wf = tf.keras.layers.Flatten()(x_wf)
        diff = tf.keras.layers.Flatten()(diff)

        x = tf.keras.layers.Concatenate()([x_wf, diff])
        x = tf.keras.layers.Dense(768)(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.Dense(32)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(16)(x)
        x = tf.keras.layers.Dense(1, activation = "sigmoid")(x)

        model = tf.keras.Model(inputs = [input_wf, input_prev_wf, input_similarity], outputs = x)

        return model



## Similarity Criticizer: a VAE trained on encoding the images in the dataset into embeddings
class Similarity_Criticizer(tf.keras.Model):

    def __init__(self, width, height, depth, embedding_dim):
        super(Similarity_Criticizer, self).__init__()

        self.width = int(width)
        self.height = int(height)
        self.depth = int(depth)
        self.embedding_dim = int(embedding_dim)

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
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

        x = tf.keras.layers.Conv2D(4, kernel_size=(3,3), strides=(2,2), activation="relu", padding="same")(encoder_inputs)
        x = tf.keras.layers.Conv2D(8, kernel_size=(3,3), strides=(2,2), activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv2D(16, kernel_size=(3,3), strides=(2,2), activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv2D(64, kernel_size=(1,1), strides=(1,1), activation="relu", padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.embedding_dim, activation="relu")(x)

        z_mean = tf.keras.layers.Dense(self.embedding_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(self.embedding_dim, name="z_log_var")(x)
        z = self.Sampling()([z_mean, z_log_var])

        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder
    
    def get_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.embedding_dim,))

        x = tf.keras.layers.Dense(self.width//16*self.height//16 * (self.embedding_dim//(self.width//16*self.height//16)+1), activation="relu")(latent_inputs)
        x = tf.keras.layers.Reshape((self.width // 16, self.height // 16, self.embedding_dim // (self.width // 16 * self.height // 16) + 1))(x)

        x = tf.keras.layers.Conv2DTranspose(8, kernel_size=(3,3), strides=(2,2), padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(4, kernel_size=(3,3), strides=(2,2), padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(2, kernel_size=(3,3), strides=(2,2), padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(1, kernel_size=(3,3), strides=(2,2), padding="same")(x)
        decoder_outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=(1,1), strides=(1,1), padding="same")(x)

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
            reconstruction_loss = tf.keras.losses.MeanSquaredError()(data, reconstruction)
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



class ConditionalDCGAN(tf.keras.Model):
    
    def __init__(self, batch_size, similarity_criticizer, global_sim_embeddings):
        super(ConditionalDCGAN, self).__init__()
        self.generator = Generator().generator
        self.discriminator = Discriminator().discriminator
        self.similarity_criticizer = similarity_criticizer
        self.batch_size = batch_size
        ## initialize and update at each epoch end
        self.predicted_imgs = tf.zeros((270, 112, 112, 1))
        self.fake_sim_embeddings = self.cosine_distance()
        ## true image differences for mitigation of mode collapse
        self.real_sim_embeddings = global_sim_embeddings

    def weighted_pixelwise_crossentropy(self, y_true, y_pred):
        ## calculate true positives
        y_pred = K.flatten(tf.cast(y_pred, tf.float32))
        y_true = K.flatten(tf.cast(y_true, tf.float32))
        ones = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.float32))
        zeros = tf.reduce_sum(tf.cast(tf.equal(y_true, 0), tf.float32))
        weight_vector = y_true * (1 - ones/(ones + zeros)) + (1 - y_true) * (1 - zeros/(ones + zeros))
        ## weighted BCE loss
        b_ce = K.binary_crossentropy(y_true, y_pred)
        weighted_b_ce = weight_vector * b_ce
        ## Dice loss
        smooth = 0.01
        intersection = tf.reduce_sum(y_pred * y_true)                           
        dice = (2. * intersection + smooth) / (tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) + smooth)
        return K.mean(weighted_b_ce) * 0.05 + (1 - dice) * (1 - 0.05)
        
    def discriminator_loss(self, real_output, fake_output):
        ## ability to differentiate between complepte "real" against inferred "real" 
        real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real_output), real_output)   
        ## ability to differentiate between complepte "fake" against inferred "fake" 
        fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(fake_output), fake_output)  
        total_loss = real_loss + fake_loss
        return total_loss * 100

    def generator_loss(self, fake_output, fake_img, real_img):
        adversarial_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(fake_output), fake_output)
        weighted_loss = self.weighted_pixelwise_crossentropy(real_img, fake_img)
        return adversarial_loss * 0.05 + weighted_loss * (1 - 0.05)
    
    def scoring(self, real_img, fake_img):
        fake_img = fake_img.numpy()
        real_img = real_img.numpy()
        fake_img = fake_img.astype(int)
        real_img = real_img.astype(int)
        acc = accuracy_score(real_img.reshape(-1, 1), fake_img.reshape(-1, 1))
        jac = jaccard_score(real_img.reshape(-1, 1), fake_img.reshape(-1, 1))
        return [acc, jac]
    
    def compile(self, d_optimizer, g_optimizer):
        super(ConditionalDCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = self.discriminator_loss
        self.g_loss_fn = self.generator_loss

    def cosine_distance(self):
        embeddings = self.similarity_criticizer.predict(self.predicted_imgs, batch_size=10)
        sim_embeddings = np.array(1 - cosine_similarity(embeddings, embeddings))
        return sim_embeddings.reshape(1, sim_embeddings.shape[0], sim_embeddings.shape[0], 1)

    def train_step(self, data):
        batch_size = tf.shape(data[0])[0]
        features = data[0]
        images = data[1]
        prev_labels = data[2]
        fake_sim_embeddings = np.concatenate([
                                                [self.fake_sim_embeddings] * self.batch_size
                                            ], axis=0).reshape(self.batch_size, 270, 270, 1)
        real_sim_embeddings = self.real_sim_embeddings

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            noise = tf.random.normal([batch_size, tf.shape(data[0])[1], 56, 56, 50])
            generated_images = self.generator([noise, features, prev_labels])
            real_output = self.discriminator([images, prev_labels, real_sim_embeddings])
            fake_output = self.discriminator([generated_images, prev_labels, fake_sim_embeddings])
            
            d_loss = self.d_loss_fn(real_output, fake_output)
            g_loss = self.g_loss_fn(fake_output, generated_images, images)

        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_weights))
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_weights))

        scores = self.scoring(images, generated_images)

        return {"d_loss": d_loss, "g_loss": g_loss, "accuracy": scores[0], "IOU": scores[1]}
