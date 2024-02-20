"""
@author: Skye Cui
@file: train_cpu.py
@time: 2021/2/20 16:35
@description: 
"""
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.realpath(".."))
sys.path.append('/home/dl/Desktop/ENSO_MC/')

os.environ["LOGURU_INFO_COLOR"] = "<green>"

import time
import re
import tensorflow as tf
import random
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from loguru import logger
from progress.spinner import MoonSpinner
from data_process.transfer.input import *
# from multivar.CERA.input import *
from model import UConvlstm
# from model import UConvlstm
from component.loss import Loss

from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


train_dataset, test_dataset = train_input_fn()
optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr)
model = UConvlstm(hp, hp.num_predictor)
model_loss = Loss(model)

checkpoint_file = hp.ckpt
if checkpoint_file == '':
    checkpoint_file = 'uconvlstm-ckp_0'
else:
    model.load_weights(f'{hp.single_gpu_model_dir}/{checkpoint_file}')
    print("model loaded!")

logger.add(f"{hp.logdir}/train.log", enqueue=True)

best_loss = float('inf')

for epoch in range(hp.num_epochs):
    total_epoch = int(re.findall("\d+", checkpoint_file)[-1])
    # print(re.findall("\d+", checkpoint_file))
    checkpoint_file = checkpoint_file.replace(f'_{total_epoch}', f'_{total_epoch + 1}')

    teacher_forcing_ratio = np.maximum(0, 1 - epoch * 0.01)

    for step, (input_tensor, target_tensor) in enumerate(train_dataset):
        start = time.process_time()
        loss_l1_train = 0
        loss_l2_train = 0
        loss_train = 0

        use_teacher_forcing = True if random.random(
        ) < teacher_forcing_ratio else False

        with tf.GradientTape() as tape:
            for i in range(hp.out_seqlen):
                y_i = model(input_tensor, training=True)
                target = target_tensor[:, i, ...]
                loss_l2, loss_l1, loss = model_loss([y_i, target])
                loss_l1_train += loss_l1
                loss_l2_train += loss_l2
                loss_train += loss

                if use_teacher_forcing:
                    y_in = target
                else:
                    y_in = y_i
                
                input_tensor = tf.concat((input_tensor[:, 1:, ...], tf.expand_dims(y_in, 1)), axis=1)
        grads = tape.gradient(loss_train, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 50 == 0:
            elapsed = (time.process_time() - start)
            template = ("step {} loss is {:1.5f}, "
                        "loss l2 is {:1.5f}, "
                        "loss l1 is {:1.5f}."
                        "({:1.2f}s/step)")
            logger.info(
                template.format(step, loss_train, loss_l2_train, loss_l1_train, elapsed))

    if epoch % hp.num_epoch_record == 0:
        loss_test = 0
        loss_l2_test = 0
        loss_l1_test = 0
        count = 0
        spinner = MoonSpinner('Testing ')
        for step, (input_test, target_test) in enumerate(test_dataset):
            step_loss_test = 0
            step_loss_l2_test = 0
            step_loss_l1_test = 0
            for i in range(hp.out_seqlen):
                yy_i = model(input_test, training=False)
                loss_l2, loss_l1, loss = model_loss([yy_i, target_test[:, i, ...]])
                step_loss_l2_test += loss_l2.numpy()
                step_loss_l1_test += loss_l1.numpy()
                step_loss_test += loss.numpy()
                input_test = tf.concat((input_test[:, 1:, ...], tf.expand_dims(yy_i, 1)), axis=1)
            count += 1
            loss_test += step_loss_test
            loss_l2_test += step_loss_l2_test
            loss_l1_test += step_loss_l1_test
            spinner.next()
        spinner.finish()
        logger.info("TEST COMPLETE!")
        template = ("TEST DATASET STATISTICS: "
                    "loss is {:1.5f}, "
                    "loss l2 is {:1.5f}, "
                    "loss l1 is {:1.5f}.")
        logger.info(
            template.format(loss_test / count, loss_l2_test / count, loss_l1_test / count))

        if loss_test < best_loss:
            best_loss = loss_test
            model.save_weights(f'{hp.single_gpu_model_dir}/{checkpoint_file}', save_format='tf')
            logger.info("Saved checkpoint_file {}".format(checkpoint_file))
