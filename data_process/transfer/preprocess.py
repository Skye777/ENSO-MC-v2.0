"""
@author: Skye Cui
@file: preprocess_godas.py
@time: 2021/6/8 17:22
@description: 
"""
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.realpath(".."))
sys.path.append('/home/dl/Desktop/ENSO_MC/')

import numpy as np
import tensorflow as tf
import netCDF4 as nc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from progress.bar import PixelBar

from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


# ---------- Helpers ----------
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# ---------- Prepare Data ----------
def parse_transfer_npz_data():
    height = hp.height
    width = hp.width

    sst = np.load(f"npz_data/sst.npz")['sst']
    t300 = np.load(f"npz_data/t300.npz")['t300']
    uwind = np.load(f"npz_data/uwind.npz")['u10']
    vwind = np.load(f"npz_data/vwind.npz")['v10']

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, height * width))), (-1, height, width))
    t300 = np.reshape(scaler.fit_transform(np.reshape(t300, (-1, height * width))), (-1, height, width))
    uwind = np.reshape(scaler.fit_transform(np.reshape(uwind, (-1, height * width))), (-1, height, width))
    vwind = np.reshape(scaler.fit_transform(np.reshape(vwind, (-1, height * width))), (-1, height, width))

    data = []
    target = []
    for i in range(sst.shape[0] - hp.in_seqlen + 1 - hp.out_seqlen):
        data.append({
            'sst': sst[i:i + hp.in_seqlen].astype(np.float32),
            't300': t300[i:i + hp.in_seqlen].astype(np.float32),
            'u10': uwind[i:i + hp.in_seqlen].astype(np.float32),
            'v10': vwind[i:i + hp.in_seqlen].astype(np.float32)
        })

        target_start = i + hp.in_seqlen
        target.append({
            'sst':
            sst[target_start:target_start + hp.out_seqlen].astype(np.float32),
            't300':
            t300[target_start:target_start + hp.out_seqlen].astype(np.float32),
            'u10':
            uwind[target_start:target_start + hp.out_seqlen].astype(np.float32),
            'v10':
            vwind[target_start:target_start + hp.out_seqlen].astype(np.float32),
        })

    print(len(data), len(target))
    return data, target


# ---------- IO ----------
def write_records(data, filename):
    series = data[0]
    target = data[1]
    writer = tf.io.TFRecordWriter(f"tfRecords/{filename}")

    bar = PixelBar(r'Generating', max=len(data), suffix='%(percent)d%%')
    for s, t in zip(series, target):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'input_sst': _bytes_feature(s['sst'].tobytes()),
                'input_t300': _bytes_feature(s['t300'].tobytes()),
                'input_u10': _bytes_feature(s['u10'].tobytes()),
                'input_v10': _bytes_feature(s['v10'].tobytes()),
                'output_sst': _bytes_feature(t['sst'].tobytes()),
                'output_t300': _bytes_feature(t['t300'].tobytes()),
                'output_u10': _bytes_feature(t['u10'].tobytes()),
                'output_v10': _bytes_feature(t['v10'].tobytes()),
            }))
        writer.write(example.SerializeToString())
        bar.next()
    writer.close()
    bar.finish()


# ---------- Go! ----------
if __name__ == "__main__":
    print("Parsing raw data...")
    transfer_data, transfer_target = parse_transfer_npz_data()
    train_data, train_target = transfer_data[:312], transfer_target[:312]
    test_data, test_target = transfer_data[360:], transfer_data[360:]
    print(len(train_data), len(train_target))
    print(len(test_data), len(test_target))
    print("Writing TF Records to file...")
    write_records((train_data, train_target), "train.tfrecords")
    write_records((test_data, test_target), "test.tfrecords")

    print("Done!")
