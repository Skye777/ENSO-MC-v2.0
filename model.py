"""
@author: Skye Cui
@file: model.py
@time: 2021/2/23 15:55
@description: 
"""
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.realpath(".."))
sys.path.append('/home/dl/Desktop/ENSO_MC/')

import tensorflow as tf
from iterate.layers import ConvlstmMaxPoolBlock, ConvTransBlock, DoubleConv


def UConvlstm(hp, num_predictor):
    filter_params = [8, 16, 32, 64, 128]

    inp = tf.keras.Input(shape=(hp.in_seqlen, hp.height, hp.width, num_predictor))
    out, map0 = DoubleConv(out_channels=filter_params[0], mid_channels=filter_params[0],
                           t=hp.in_seqlen, h=80, w=160, c=filter_params[0], k=hp.kunit)(inp)    # map0 (b, 80, 160, 8) out(b, t, 80, 160, 8)
    out, map1 = ConvlstmMaxPoolBlock(filters=filter_params[1], kernel_size=3, pool_size=2, strides=2,
                                                   t=hp.in_seqlen, h=40, w=80, c=filter_params[1], k=hp.kunit)(out)  # map1 (b, 40, 80, 16) out(b, t, 40, 80, 16)
    out, map2 = ConvlstmMaxPoolBlock(filters=filter_params[2], kernel_size=3, pool_size=2, strides=2,
                                                   t=hp.in_seqlen, h=20, w=40, c=filter_params[2], k=hp.kunit)(out)  # map2 (b, 20, 40, 32) out(b, t, 20, 40, 32)
    out, map3 = ConvlstmMaxPoolBlock(filters=filter_params[3], kernel_size=3, pool_size=2, strides=2,
                                                   t=hp.in_seqlen, h=10, w=20, c=filter_params[3], k=hp.kunit)(out)  # map3 (b, 10, 20, 64) out(b, t, 10, 20, 64)
    out = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(map3)
    # D1
    conv1_1 = tf.keras.layers.Conv2D(filter_params[4], 3, activation=tf.keras.layers.LeakyReLU(), padding='same')(out)
    conv1_2 = tf.keras.layers.Conv2D(filter_params[4], 3, activation=tf.keras.layers.LeakyReLU(), padding='same')(conv1_1)
    # D2
    conv2_1 = tf.keras.layers.Conv2D(filter_params[4], 3, activation=tf.keras.layers.LeakyReLU(), padding='same')(conv1_2)
    conv2_2 = tf.keras.layers.Conv2D(filter_params[4], 3, activation=tf.keras.layers.LeakyReLU(), padding='same')(conv2_1)
    # D3
    merge_dense = tf.keras.layers.concatenate([conv2_2, conv1_2], axis=3)
    conv3_1 = tf.keras.layers.Conv2D(filter_params[4], 3, activation=tf.keras.layers.LeakyReLU(), padding='same')(merge_dense)
    conv3_2 = tf.keras.layers.Conv2D(filter_params[4], 3, activation=tf.keras.layers.LeakyReLU(), padding='same')(conv3_1)   # (b, 5, 10, 128)

    # (b, 5, 10, 128) --> (b, 5, 10, 64)
    mid_state = tf.keras.layers.Conv2D(filter_params[3], 3, activation=tf.keras.layers.LeakyReLU(), padding='same')(conv3_2)

    out = ConvTransBlock(filters=filter_params[2], kernel_size=3, up_size=2)([mid_state, map3])
    out = ConvTransBlock(filters=filter_params[1], kernel_size=3, up_size=2)([out, map2])
    out = ConvTransBlock(filters=filter_params[0], kernel_size=3, up_size=2)([out, map1])
    out = ConvTransBlock(filters=filter_params[0], kernel_size=3, up_size=2)([out, map0])

    out = tf.keras.layers.Conv2D(filter_params[1], 3, activation=tf.keras.layers.LeakyReLU(), padding='same')(out)
    out = tf.keras.layers.Conv2D(filter_params[1], 3, activation=tf.keras.layers.LeakyReLU(), padding='same')(out)
    out = tf.keras.layers.Conv2D(hp.num_output*2, 3, activation=tf.keras.layers.LeakyReLU(), padding='same')(out)
    out = tf.keras.layers.Conv2D(hp.num_output, 1, activation='sigmoid')(out)
    
    model = tf.keras.Model(inp, out)
    return model


if __name__ == '__main__':
    from hparams import Hparams

    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()

    inputs = tf.random.uniform(shape=(hp.batch_size, hp.in_seqlen, hp.height, hp.width, hp.num_predictor))
    model = UConvlstm(hp, num_predictor=4)
    out = model(inputs)
    print(out.shape)
    print(model.summary())
