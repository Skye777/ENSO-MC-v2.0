"""
@author: Skye Cui
@file: layers.py
@time: 2021/2/22 13:43
@description: 
"""
import tensorflow as tf


class ConvAttention(tf.keras.layers.Layer):

    def __init__(self, l, h, w, c, k):
        super(ConvAttention, self).__init__()
        self.reshape = tf.keras.layers.Reshape((l, w * h * c))
        self.layer1 = tf.keras.layers.Dense(units=k, activation='tanh')
        self.layer2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=None):
        outputs = self.layer2(self.layer1(self.reshape(inputs)))
        outputs = tf.nn.softmax(outputs, axis=-2)
        return outputs


class WeightedSumBlock(tf.keras.layers.Layer):

    def __init__(self, l, h, w, c):
        super(WeightedSumBlock, self).__init__()
        self.l = l
        self.add = tf.keras.layers.Add()
        self.reshape1 = tf.keras.layers.Reshape((l, w * h * c))
        self.reshape2 = tf.keras.layers.Reshape((h, w, c))

    def call(self, inputs, training=None):
        # alpha (b, t, 1)
        inputs, alpha = inputs
        inputs = self.reshape1(inputs)  # (batch, time, feature)
        info = tf.multiply(alpha, inputs)  # (batch, time, feature)
        # time*(batch, 1, feature)
        info = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=self.l, axis=-2))(info)
        outputs = tf.keras.layers.add(info)  # (batch, 1, feature)
        outputs = self.reshape2(outputs)
        return outputs


class DoubleConv(tf.keras.layers.Layer):

    def __init__(self, out_channels, mid_channels, t, h, w, c, k):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = tf.keras.Sequential()
        self.double_conv.add(tf.keras.layers.ConvLSTM2D(filters=mid_channels,
                                       kernel_size=3,
                                       padding='same',
                                       return_sequences=True))
        self.double_conv.add(tf.keras.layers.LayerNormalization())
        self.double_conv.add(tf.keras.layers.LeakyReLU())
        self.double_conv.add(tf.keras.layers.ConvLSTM2D(filters=out_channels,
                                       kernel_size=3,
                                       padding='same',
                                       return_sequences=True))
        self.double_conv.add(tf.keras.layers.LayerNormalization())
        self.double_conv.add(tf.keras.layers.LeakyReLU())

        self.alpha = ConvAttention(t, h, w, c, k)
        self.get_feature_maps = WeightedSumBlock(t, h, w, c)

    def call(self, inputs):
        out = self.double_conv(inputs)
        alpha = self.alpha(out)
        skip_layer_feature_map = self.get_feature_maps([out, alpha])
        return out, skip_layer_feature_map


class ConvlstmMaxPoolBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, pool_size, strides, t, h, w, c, k):
        super(ConvlstmMaxPoolBlock, self).__init__()
        self.convlstm = tf.keras.layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                   return_sequences=True, return_state=True)
        self.max_pool = tf.keras.layers.MaxPool3D(pool_size=(1, pool_size, pool_size), strides=(1, strides, strides))
        self.norm = tf.keras.layers.LayerNormalization()
        self.act = tf.keras.layers.LeakyReLU()
        self.alpha = ConvAttention(t, h, w, c, k)
        self.get_feature_maps = WeightedSumBlock(t, h, w, c)

    def call(self, inputs, skip_layer=None):
        output = self.max_pool(inputs)
        out, _, _ = self.convlstm(output)
        bn_out = self.norm(out)
        act_out = self.act(bn_out)
        alpha = self.alpha(act_out)
        skip_layer_feature_map = self.get_feature_maps([act_out, alpha])

        return act_out, skip_layer_feature_map


class ConvTransBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, up_size):
        super(ConvTransBlock, self).__init__()

        self.conv_trans = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            padding='same')
        self.up_sampling2d = tf.keras.layers.UpSampling2D(size=(up_size,
                                                                up_size))
        self.norm = tf.keras.layers.LayerNormalization()
        self.act = tf.keras.layers.LeakyReLU()

    def call(self, inputs, training=None):
        x, skip_layer = inputs

        up_out = self.up_sampling2d(x)
        merge = tf.keras.layers.Concatenate()([skip_layer, up_out])
        deconv_out = self.conv_trans(merge)
        bn_out = self.norm(deconv_out)
        act_out = self.act(bn_out)
        return act_out


if __name__ == '__main__':
    from hparams import Hparams

    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()

    # inputs = tf.random.uniform(shape=(hp.batch_size, hp.out_seqlen, hp.height, hp.width, hp.num_predictor))
    # encoder = EnConvlstm(seq_len=hp.in_seqlen)
    # decoder = DeConvlstm(strategy=hp.strategy, out_seqlen=hp.out_seqlen)
    # mid_state, hidden_states, skip_layers = encoder(inputs)
    # output = decoder([mid_state, hidden_states, skip_layers])
    # print(output.shape)
