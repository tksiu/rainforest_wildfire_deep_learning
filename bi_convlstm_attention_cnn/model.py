import tensorflow as tf


###  strides parameter for each convolution

def find_strides(self, n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors



###  align the output embedding dimension of features convolved from the regional weather window and the local fire predicting grids (5 degree x 5 degree Lat/Lon)

def grid_window_alignment_operation(self, x2, x3):

        if self.weather_window_height == self.weather_window_width:

            window_factor = int(self.weather_window_height / self.land_grid_height)
            window_encoders = self.find_strides(window_factor) + [1]

            if self.weather_window_height > self.land_grid_height:
                pad_window_factor = self.land_grid_height - self.weather_window_height % self.land_grid_height
                pad_window_factor = int(pad_window_factor)
            else:
                pad_window_factor = self.land_grid_height - self.weather_window_height
                pad_window_factor = int(pad_window_factor)
            
            if pad_window_factor > 0:
                x2 = tf.keras.layers.ZeroPadding3D(padding=(0, int(0.5 * pad_window_factor), int(0.5 * pad_window_factor)))(x2)
                x3 = tf.keras.layers.ZeroPadding3D(padding=(0, int(0.5 * pad_window_factor), int(0.5 * pad_window_factor)))(x3)

            for w in window_encoders:

                self.convlstm_2 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(5, 5), strides=(w, w), 
                                                             padding='same', return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
                self.bi_convlstm_2 = tf.keras.layers.Bidirectional(self.convlstm_2)
                x2 = self.bi_convlstm_2(x2)
                x2 = tf.keras.layers.BatchNormalization()(x2)

                self.convlstm_3 = tf.keras.layers.ConvLSTM2D(8, kernel_size=(5, 5), strides=(w, w), 
                                                             padding='same', return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
                self.bi_convlstm_3 = tf.keras.layers.Bidirectional(self.convlstm_3)
                x3 = self.bi_convlstm_3(self.x3)
                x3 = tf.keras.layers.BatchNormalization()(x3)

        elif self.weather_window_height != self.weather_window_width:

            window_factor_height = int(self.weather_window_height / self.land_grid_height)
            window_encoders_height = self.find_strides(window_factor_height) + [1]

            if self.weather_window_height > self.land_grid_height:
                pad_window_height = self.land_grid_height - self.weather_window_height % self.land_grid_height
                pad_window_height = int(pad_window_height)
            else:
                pad_window_height = self.land_grid_height - self.weather_window_height
                pad_window_height = int(pad_window_height)

            window_factor_width = int(self.weather_window_width / self.land_grid_width)
            window_encoders_width = self.find_strides(window_factor_width) + [1]

            if self.weather_window_width > self.land_grid_width:
                pad_window_width = self.land_grid_width - self.weather_window_width % self.land_grid_width
                pad_window_width = int(pad_window_width)
            else:
                pad_window_width = self.land_grid_width - self.weather_window_width
                pad_window_width = int(pad_window_width)

            if pad_window_height > 0 or pad_window_width > 0:
                x2 = tf.keras.layers.ZeroPadding3D(padding=(0, int(0.5 * pad_window_height), int(0.5 * pad_window_width)))(x2)
                x3 = tf.keras.layers.ZeroPadding3D(padding=(0, int(0.5 * pad_window_height), int(0.5 * pad_window_width)))(x3)

            if len(window_encoders_height) > len(window_encoders_width):
                window_encoders_width = window_encoders_width + [1] * (len(window_encoders_height) - len(window_encoders_width))
            elif len(window_encoders_width) > len(window_encoders_height):
                window_encoders_height = window_encoders_height + [1] * (len(window_encoders_width) - len(window_encoders_height))

            window_encoders = [window_encoders_height, window_encoders_width]
            len_window_encoders = len(window_encoders[0])

            for w in range(len_window_encoders):

                self.convlstm_2 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(5, 5), strides=(window_encoders[0][w], window_encoders[1][w]), 
                                                             padding='same', return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
                self.bi_convlstm_2 = tf.keras.layers.Bidirectional(self.convlstm_2)
                self.x2 = self.bi_convlstm_2(self.x2)
                self.convlstm_2 = tf.keras.layers.BatchNormalization()(self.x2)

                self.convlstm_3 = tf.keras.layers.ConvLSTM2D(8, kernel_size=(5, 5), strides=(window_encoders[0][w], window_encoders[1][w]), 
                                                             padding='same', return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
                self.bi_convlstm_3 = tf.keras.layers.Bidirectional(self.convlstm_3)
                self.x3 = self.bi_convlstm_3(self.x3)
                self.convlstm_3 = tf.keras.layers.BatchNormalization()(self.x3)
        
        return [x2, x3]



def encoder(self, num_encoder_layers):

    """ Construct the input layer with no definite frame size. """
    self.channel_1 = tf.keras.layers.Input(shape=(None, self.land_grid_height, self.land_grid_width, 
                                                    self.land_grid_features * self.neighbour_window))
    self.channel_2 = tf.keras.layers.Input(shape=(None, self.weather_window_height * 4, self.weather_window_width * 4, 9))
    self.channel_3 = tf.keras.layers.Input(shape=(None, self.weather_window_height, self.weather_window_width, 4))
    
    """ Bidirectional Convolutional LSTM """
    self.convlstm_1 = tf.keras.layers.ConvLSTM2D(128, kernel_size=(5, 5), strides=(1, 1), padding="same", 
                                                    return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
    self.bi_convlstm_1 = tf.keras.layers.Bidirectional(self.convlstm_1)
    self.x1 = self.bi_convlstm_1(self.channel_1)
    self.x1 = tf.keras.layers.BatchNormalization()(self.x1)

    for n in range(num_encoder_layers - 1):
        self.convlstm_1 = tf.keras.layers.ConvLSTM2D(64, kernel_size=(5, 5), strides=(1, 1), padding="same", 
                                                        return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
        self.bi_convlstm_1 = tf.keras.layers.Bidirectional(self.convlstm_1)
        self.x1 = self.bi_convlstm_1(self.x1)
        self.x1 = tf.keras.layers.BatchNormalization()(self.x1)

    self.x2 = self.channel_2
    self.x3 = self.channel_3

    for m in range(2):
        self.convlstm_2 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(5, 5), strides=(2, 2), padding='same', 
                                                    return_sequences=True, dropout=0.1, recurrent_dropout=0.02)
        self.bi_convlstm_2 = tf.keras.layers.Bidirectional(self.convlstm_2)
        self.x2 = self.bi_convlstm_2(self.x2)
        self.x2 = tf.keras.layers.BatchNormalization()(self.x2)

    window_encoded_output = self.grid_window_alignment_operation(self.x2, self.x3)
    self.x2 = window_encoded_output[0]
    self.x3 = window_encoded_output[1]

    self.x = tf.keras.layers.Concatenate(axis=4)([self.x1, self.x2, self.x3])

    return self.x



def multihead_attn(self):

    """ Attention on Temporal Sequence """
    self.att = tf.keras.layers.MultiHeadAttention(num_heads = self.lag_window, 
                                                    key_dim = self.lag_window, 
                                                    attention_axes=(1,2,3))(self.x, self.x)
    self.reshape_att = tf.keras.layers.Reshape(target_shape=(self.land_grid_height, 
                                                                self.land_grid_width, 
                                                                self.lag_window * (128+32+16)))(self.att)
    
    return self.reshape_att



def concat_static_features(self):

    self.channel_4 = tf.keras.layers.Input(shape=(self.img_height, self.img_width, 3))
    self.channel_5 = tf.keras.layers.Input(shape=(int(self.img_height / 4), int(self.img_width / 4), 3))

    self.conv_enc_4 = tf.keras.layers.Conv2D(64, 5, activation="relu", strides=1, padding="same")(self.channel_4)
    self.conv_enc_4 = tf.keras.layers.ZeroPadding2D(padding=(int(0.5 * (self.land_grid_height - self.img_height % self.land_grid_height)), 
                                                                int(0.5 * (self.land_grid_width - self.img_width % self.land_grid_width))
                                                            ))(self.conv_enc_4)
    x_z1_stride = (self.img_height + self.land_grid_height - self.img_height % self.land_grid_height) // self.land_grid_height
    self.z1 = tf.keras.layers.Conv2D(64, 5, activation="relu", padding="same",
                                        strides = (x_z1_stride, x_z1_stride)
                                        )(self.conv_enc_4)
    self.z1 = tf.keras.layers.BatchNormalization()(self.z1)

    self.conv_enc_5 = tf.keras.layers.Conv2D(64, 5, activation="relu", strides=1, padding="same")(self.channel_5)
    self.conv_enc_5 = tf.keras.layers.ZeroPadding2D(padding=(int(0.5 * (self.land_grid_height - (self.img_height / 4) % self.land_grid_height)), 
                                                                int(0.5 * (self.land_grid_width - (self.img_width / 4) % self.land_grid_width))
                                                            ))(self.conv_enc_5)
    x_z2_stride = int(((self.img_height / 4) + self.land_grid_height - (self.img_height / 4) % self.land_grid_height) // self.land_grid_height)                                                        
    self.z2 = tf.keras.layers.Conv2D(64, 5, activation="relu", padding="same",
                                        strides = (x_z2_stride, x_z2_stride)
                                        )(self.conv_enc_5)
    self.z2 = tf.keras.layers.BatchNormalization()(self.z2)

    self.reshape_att = tf.keras.layers.Concatenate(axis=3)([self.reshape_att, self.z1, self.z2])

    return self.reshape_att



###  deconvolution module for flexibly adapting the output image resolution, and progressively upsampling by a factor of 2 per each layer

def decoder(self, num_decoder_layers, fixed_intermediate=False):

    if fixed_intermediate:

        self.deconv = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=1, padding='same')(self.reshape_att)

        for n in range(num_decoder_layers - 1):
            self.deconv = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=1, padding='same')(self.deconv)

        self.outcome = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 1), strides=1, padding='same')(self.deconv)

    else:

        if self.output_img_height > self.land_grid_height and self.output_img_width > self.land_grid_width:

            required_strides = (self.output_img_height - self.output_img_height % self.land_grid_height) // self.land_grid_height
            strides = self.find_strides(required_strides) + [1]
            self.pad_att = tf.keras.layers.ZeroPadding2D(padding=(int(0.5 * (self.output_img_height % self.land_grid_height)), 
                                                                    int(0.5 * (self.output_img_width % self.land_grid_width))
                                                                    ))(self.reshape_att)
            
            if num_decoder_layers > len(strides):
                if num_decoder_layers < 6:
                    start_filter = 128
                else:
                    start_filter = 2 ** num_decoder_layers
            else:
                if len(strides) < 6:
                    start_filter = 128
                else:
                    start_filter = 2 ** len(strides)

            self.deconv = self.pad_att
            for s in strides:
                self.deconv = tf.keras.layers.Conv2DTranspose(filters = start_filter/2, kernel_size=(3, 3), strides = s, padding='same')(self.deconv)
                start_filter = start_filter // 2
            if num_decoder_layers > len(strides):
                for r in range(num_decoder_layers - len(strides)):
                    self.deconv = tf.keras.layers.Conv2DTranspose(filters = start_filter, kernel_size=(3, 3), strides = 1, padding='same')(self.deconv)
            self.outcome = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 1), strides=1, padding='same')(self.deconv)

        elif self.output_img_height < self.land_grid_height and self.output_img_width < self.land_grid_width:

            self.pad_att = tf.keras.layers.ZeroPadding2D(padding=(int(0.5 * (self.output_img_height - self.land_grid_height % self.output_img_height)), 
                                                                    int(0.5 * (self.output_img_width - self.land_grid_width % self.output_img_width))
                                                                    ))(self.reshape_att)
            x_stride_att = (self.land_grid_height + self.output_img_height - self.land_grid_height % self.output_img_height) // self.output_img_height
            y_stride_att = (self.land_grid_height + self.output_img_height - self.land_grid_height % self.output_img_height) // self.output_img_height
            self.pad_att = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same",
                                                    strides = (x_stride_att, y_stride_att)
                                                    )(self.pad_att)
            
            if num_decoder_layers < 6:
                start_filter = 128
            else:
                start_filter = 2 ** num_decoder_layers
            
            self.deconv = tf.keras.layers.Conv2DTranspose(filters = start_filter, kernel_size=(3, 3), strides=1, padding='same')(self.reshape_att)

            for d in range(num_decoder_layers - 1):
                self.deconv = tf.keras.layers.Conv2DTranspose(filters = start_filter/2, kernel_size=(3, 3), strides=1, padding='same')(self.deconv)
                start_filter = start_filter // 2

            self.outcome = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 1), strides=1, padding='same')(self.deconv)

    return self.outcome






class BiConvLSTM_Attention_Model(tf.keras.Model):
    """Convolutional LSTM with Attention on previous time/states"""
    
    def __init__(self, mode,
                       img_height, img_width, 
                       output_img_height, output_img_width, 
                       lag_window, neighbour_window,
                       land_grid_height, land_grid_width, land_grid_features,
                       weather_window_height, weather_window_width):
      
        super(BiConvLSTM_Attention_Model, self).__init__()

        self.img_height, self.img_width = img_height, img_width
        self.output_img_height, self.output_img_width = output_img_height, output_img_width
        self.land_grid_height, self.land_grid_width = land_grid_height, land_grid_width
        self.weather_window_height, self.weather_window_width = weather_window_height, weather_window_width
        self.lag_window = lag_window
        self.neighbour_window = neighbour_window
        self.land_grid_features = land_grid_features
        self.mode = mode

        self.find_strides = find_strides
        self.grid_window_alignment_operation = grid_window_alignment_operation
        self.encoder = encoder
        self.multihead_attn = multihead_attn
        self.concat_static_features = concat_static_features
        self.decoder = decoder
        

        ## parameter check
        if mode not in ["monthly","daily"]:
            raise ValueError("Invalid value, mode should be either 'monthly' or 'daily'.")
        if img_height / 4 < land_grid_height or img_width / 4 < land_grid_width:
            raise ValueError("Image dimension is too small, should be greater than four times of the land grid dimension.")
        if land_grid_height != land_grid_width:
            raise ValueError("Non-square land grid dimension is detected.")
    

    def get_model(self, num_encoder_layers, num_decoder_layers, fixed_decoder_intermediate):

        """ Construct the input layer with no definite frame size. """
        self.x = self.encoder(num_encoder_layers)
        
        """ Attention on Temporal Sequence """
        self.reshape_att = self.multihead_attn()

        """ Long-term features """
        if self.mode == "monthly":
            self.reshape_att = self.concat_static_features()

        """ Deconvolution """
        self.outcome = self.decoder(num_decoder_layers, fixed_decoder_intermediate)

        """Activation"""
        self.outcome = tf.keras.layers.BatchNormalization()(self.outcome)
        self.outcome = tf.keras.layers.Activation('sigmoid')(self.outcome)
        
        if self.mode == "monthly":
            self.model = tf.keras.Model(inputs = [self.channel_1, self.channel_2, self.channel_3, self.channel_4, self.channel_5], 
                                        outputs = self.outcome)
        else:
            self.model = tf.keras.Model(inputs = [self.channel_1, self.channel_2, self.channel_3], 
                                        outputs = self.outcome)
            
        return self.model
    

