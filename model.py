import go
import coords
import tensorflow as tf
import numpy as np
import time
import player

class ValueNet():
    def __init__(self):
        self.conv1 = tf.keras.layers.Conv2D(100, 3, padding='same',activation='relu', )
        self.conv2 = tf.keras.layers.Conv2D(100, 3, padding='same',activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(1, 3, padding='same', activation='relu')
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.norm3 = tf.keras.layers.BatchNormalization()


        # now the value net specifics

        self.val1 = tf.keras.layers.Dense(100, activation='relu')
        self.val2 = tf.keras.layers.Dense(1, activation='tanh')


        pass
    

    def call(self, game_state):
        #game state is a tuple of type batch_size*(9x9, caps, caps)
        
        batch= list(list(zip(*game_state)))
        board = batch[0]
        
        myCaps = batch[1]

        oppCaps = batch[2]
        #features = self.conv3(self.conv2(self.conv1(board)))
        #features = self.norm3(self.conv3(self.norm2(self.conv2(self.norm1(self.conv1(board))))))
        """
        features = tf.reshape(features, [81,])
        features = tf.concat(features, tf.convert_to_tensor([myCaps, oppCaps]))
        return self.val2(self.val1(features))
        """
        return board

white = player.Player()
black = player.Player()
myData = list(list(zip(*player.selfplay(black, white))))

myVal = ValueNet()
myVal.call(myData[0])



    




