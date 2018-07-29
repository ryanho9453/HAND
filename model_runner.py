from processing.data_reader import DataReader
from models.model_picker import ModelPicker
import tensorflow as tf
import numpy as np
import json
import os
import time
import argparse

"""
/// todo
tensorboard

"""

"""
1. prepare X, Y
2. build model , output train_op, loss, logits
3. run and get train_loss, prediction


"""


def train(restore):
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_picker = ModelPicker(config['models'])
    data_reader = DataReader(config['processing']['reader'])

    with tf.name_scope('data_in'):
        if config['models']['choose_model'] == 'conv':
            img_size = config['data_spec']['img_size']
            data_in = tf.placeholder(
                tf.uint8, [None, img_size, img_size], name='data_in')

    with tf.name_scope('label_in'):
        batch_size = config['processing']['reader']['batch_size']
        label_in = tf.placeholder(
            tf.uint8, [batch_size], name='label_in')

    with tf.Session() as sess:
        model = model_picker.pick_model()

        model_save_path = config['general']['model_save_path']
        ckpt_path = model_save_path + 'model.ckpt'
        meta_path = model_save_path + 'model.ckpt.meta'
        #
        # prepare train component 1. train_op 2. loss 3. logits
        #
        if not restore:
            train_op, loss, logits = model.build(data_in, label_in)
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()

        else:
            print('Restore saved model which stored at %s' % meta_path)
            train_op, loss, logits = model.build(data_in, label_in)
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
        #
        # record training
        #
        loss_summary = tf.summary.scalar('loss', loss)
        writer = tf.summary.FileWriter(
            config['general']['tensor_log'], sess.graph)

        """ /// step? no epoch or batch_no ? """
        total_sec = 0
        for step in range(1, config['general']['steps'] + 1):
            start_time = time.time()
            train_data, train_labels = data_reader.next_batch()
            #
            # run train_op, loss, loss_summary, logits
            # got           train_loss, loss_summ, prediction
            #
            sess.run(train_op, feed_dict={
                        data_in: train_data,
                        label_in: train_labels
                    })

            if step // 100 != 0 and step % 100 == 0 or step == 1:
                train_loss, loss_summ, prediction = sess.run(
                        [loss, loss_summary, logits],
                        feed_dict={
                            data_in: train_data,
                            label_in: train_labels
                        })

                writer.add_summary(loss_summ, step)

                saver.save(
                    sess, ckpt_path,
                    global_step=tf.train.get_global_step())
                print('Save model at step = %s' % (step))
                print('loss = %s, step = %s (%s sec)'
                      % (train_loss, step, total_sec))

                print('logits[0]')
                print(prediction[0])
                print('labels[0]')
                print(train_labels[0])

                # total_sec = 0
                # Early stop
                # need_early_stop = config['general']['early_stop']
                # early_stop_loss = config['general']['ealry_stop_loss']
                # if need_early_stop and train_loss < early_stop_loss:
                #     print('Early stop at loss %f' % train_loss)
                #     break

            total_sec += time.time() - start_time
        saver.save(sess, ckpt_path)
        print('Save model at final step = %s, spend %fs' % (step, total_sec))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model runner')
    parser.add_argument('--restore', type=bool, default=True,
                        help='Restore pre-trained model store in config')
    args = parser.parse_args()
    train(args.restore)
