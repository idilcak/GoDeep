import go
import coords
import tensorflow as tf
import numpy as np
import time
import player


class ValueNet(tf.keras.Model):
    def __init__(self):
        super(ValueNet, self).__init__()
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
        # bla bla

        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.mse = tf.keras.losses.MeanSquaredError()
        pass

    def call(self, game_state):
        # game state is a tuple of type batch_size*(9x9, caps, caps)

        batch = list(list(zip(*game_state)))
        board = tf.reshape(tf.convert_to_tensor(
            batch[0], dtype=tf.int32), [50, 9, 9, 1])

        myCaps = tf.convert_to_tensor(batch[1])

        oppCaps = tf.convert_to_tensor(batch[2])

        features = self.conv1(tf.cast(board, dtype=tf.float32))
        features = self.norm1(features)
        features = self.conv2(features)
        features = self.norm2(features)
        features = self.conv3(features)
        features = self.norm3(features)

        features = tf.reshape(tf.cast(features, dtype=tf.float32), [50, 81, ])
        other_data = tf.convert_to_tensor([myCaps, oppCaps], dtype=tf.float32)
        other_data = tf.transpose(other_data)

        features = tf.concat([features, other_data], axis=1)
        print(features.shape)
        return self.val2(self.val1(features))


def loss(model, logits, labels):
    return model.mse(labels, logits)


def train(model, data, labels):
    for start in range(0, len(data) - model.batch_size, model.batch_size):
        dataBatch = data[start:start+model.batch_size]
        labelBatch = labels[start:start+model.batch_size]
        with tf.GradientTape() as tape:
            losss = loss(model, model.call(dataBatch),
                         tf.convert_to_tensor(labelBatch))
            print('loss')
            print(losss)
        gradients = tape.gradient(losss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
    print('a')
    print(model.trainable_variables)
    pass


white = player.Player()
black = player.Player()
data, labels = player.selfplay(black, white)


myVal = ValueNet()
train(myVal, data, labels)


"""
sess=tf.compat.v1.Session()
myVal=ValueNet()
saver=tf.compat.v1.train.Saver(myVal.trainable_variables, filename="val.txt")


# train
sess.run(train(myVal, data, labels))
saver.save(sess, "val.txt")
"""
a = myVal.trainable_variables
print(a)
# print('aaaa')
# print(myVal.trainable_variables)
# print('aaaa')
sess = tf.compat.v1.Session()
# train
sess.run(a)
#saver.save(sess, "val.txt")
