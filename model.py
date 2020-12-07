import go
import coords
import tensorflow as tf
import numpy as np
import time
import player


class ValueNet():
    def __init__(self):
        self.batch_size = 50
        self.conv1 = tf.keras.layers.Conv2D(
            100, 3, padding='same', activation='relu', trainable=True, dtype=tf.float32)
        self.conv2 = tf.keras.layers.Conv2D(
            100, 3, padding='same', activation='relu',  trainable=True, dtype=tf.float32)
        self.conv3 = tf.keras.layers.Conv2D(
            1, 3, padding='same', activation='relu',  trainable=True, dtype=tf.float32)
        self.norm1 = tf.keras.layers.BatchNormalization(trainable=True)
        self.norm2 = tf.keras.layers.BatchNormalization(trainable=True)
        self.norm3 = tf.keras.layers.BatchNormalization(trainable=True)

        # now the value net specifics

        self.val1 = tf.keras.layers.Dense(
            100, activation='relu', trainable=True)
        self.val2 = tf.keras.layers.Dense(1, activation='tanh', trainable=True)
        self.trainable_variables = tf.compat.v1.trainable_variables()
        # bla bla

        self.optimizer = tf.keras.optimizers.Adam(0.001)
        pass

    def call(self, game_state):
        # game state is a tuple of type batch_size*(9x9, caps, caps)

        batch = list(list(zip(*game_state)))
        board = tf.reshape(tf.convert_to_tensor(
            batch[0], dtype=tf.int32), [50, 9, 9, 1])
        print(board)

        myCaps = tf.convert_to_tensor(batch[1])

        oppCaps = tf.convert_to_tensor(batch[2])

        features = self.conv1(tf.cast(board, dtype=tf.float32))
        features = self.norm1(features)
        features = self.conv2(features)
        features = self.norm2(features)
        features = self.conv3(features)
        features = self.norm3(features)

        features = tf.reshape(tf.cast(features, dtype=tf.float32), [50, 81, ])
        features = tf.concat(features, tf.convert_to_tensor(
            tf.transpose([myCaps, oppCaps]))
        return self.val2(self.val1(features))


def loss(logits, labels):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)


def train(model, data, labels):
    for start in range(0, len(data) - model.batch_size, model.batch_size):
        dataBatch=data[start:start+model.batch_size]
        labelBatch=labels[start:start+model.batch_size]
        with tf.GradientTape() as tape:
            losss=loss(model.call(dataBatch),
                         tf.convert_to_tensor(labelBatch))
        gradients=tape.gradient(losss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
    pass


white=player.Player()
black=player.Player()
data, labels=player.selfplay(black, white)

myVal=ValueNet()
saver=tf.compat.v1.train.Saver(myVal.trainable_variables, filename="val.txt")
sess=tf.compat.v1.Session()
# train
sess.run(train(myVal, data, labels))
saver.save(sess, "val.txt")
