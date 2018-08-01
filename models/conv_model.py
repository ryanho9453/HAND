import tensorflow as tf

"""
output 3 component , 1. train_op 2. loss 3. logits

"""


class ConvModel:
    def __init__(self, config, parameter):
        self.config = config
        self.param = parameter

        self.num_class = config['data_spec']['num_class']

        self.conv_neu_num = parameter['conv_neu_num']
        self.fc_neu_num = parameter['fc_neu_num']
        self.learn_rate = parameter['learn_rate']
        self.keep_prob = parameter['keep_prob']

    def build(self, data_in, label_in):
        """
        data_in :
            pred -- dict of softmax results
                    pred = {"result": logits }

            logits -- output layer

        logits, label_in :
            train_op -- minimize loss op

            loss -- loss node

        """

        pred, logits = self.__build_model(data_in)
        train_op, loss = self.__build_train_op(logits, label_in)
        return train_op, loss, logits

    def __build_model(self, data_in):
        """
        ( 2 conv + 1 pool ) * 2 + 1 fc_layer + output_layer

        """
        neu_num = self.conv_neu_num
        with tf.variable_scope('conv_blk'):
            conv1_1 = self.__get_conv_layer(data_in, neu_num, 'conv1_1')
            conv1_2 = self.__get_conv_layer(conv1_1, neu_num, 'conv1_2')
            pool1 = self.__get_pool_layer(conv1_2, 'pool1')
            # neu_num = neu_num * 2
            conv2_1 = self.__get_conv_layer(pool1, neu_num, 'conv2_1')
            conv2_2 = self.__get_conv_layer(conv2_1, neu_num, 'conv2_2')
            pool2 = self.__get_pool_layer(conv2_2, 'pool2')

        # fc layer
        with tf.variable_scope('dense_layer'):
            """
            flatten the last layer to fit fc, 
            with the number (img_size * img_size * num_channel)
            
            dense input shape must be 2-dimension 
            
            pool2 output 16 * 16 * 64 (conv2_2 neu_num) 
            16384 = 16 * 16 * 64
            
            """

            pool2_flat = tf.reshape(pool2, [-1, 16384])

            dense = tf.layers.dense(
                inputs=pool2_flat,
                units=self.fc_neu_num,
                activation=tf.nn.relu)

            dropout = tf.layers.dropout(
                inputs=dense, rate=self.keep_prob)

        # output layer
        with tf.variable_scope('logits_layer'):

            """ /// why reshape ? is it necessary ? """

            logits = tf.layers.dense(
                inputs=dropout, units=self.num_class)
            # logits = tf.reshape(logits, [-1, self.num_class])
            pred = {
                "result": logits
            }
        return pred, logits

    def __build_train_op(self, logits, labels):
        """
        logits.shape (batch_size, num_class) float

        labels.shape (batch_size) int64

        sparse_softmax_cross_entropy :
            label_in.shape = [batch_size]    dtype : int
                ex : [1, 2, 5, 6, 9, 0 ]

            logits.shape = [batch_size, num_class]  dtype : float

        softmax_cross_entropy :
            label_in.shape = [batch_size, num_class]   dtype : float
                ex : [[0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0]]

        """
        with tf.name_scope('loss'):
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=labels, logits=logits)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learn_rate)
            train_op = optimizer.minimize(loss=loss, name='minimize')
        return train_op, loss

    def __get_conv_layer(self, data_in, filters, name, drop=False, pad=0):

        with tf.variable_scope(name):
            """
            input.dtype must be one of (float16, bfloat16, float32, float64)
            
            adjust kernel_size if necessary 
            
            padding = 'same' ,  p = (f - 1) / 2
            
            output_size = [(n + 2p - f) / s] + 1

            n = input_size                --- 
            p = padding                   --- 1
            f = filter_size(kernel_size)  --- 3
            s = strides                   --- 1
            
            
            """
            kernel_size = [3, 3]
            conv = tf.layers.conv2d(
                inputs=data_in,
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation=tf.nn.relu6
            )
            if drop:
                conv = tf.layers.dropout(conv, rate=self.keep_prob)
            return conv

    def __get_pool_layer(self, data_in, name):
        """
        output_size = [(n - f) / s] + 1

        f = filter_size(pool_size)  --- 2
        s = strides                 --- 2

        """
        return tf.layers.max_pooling2d(
            inputs=data_in, pool_size=[2, 2], strides=2, name=name)
