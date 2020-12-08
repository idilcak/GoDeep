import go
import coords
import tensorflow as tf
import numpy as np
import time
import player
import TreeSearch

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
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



        self.filter1 = tf.Variable(tf.random.normal([3, 3, 1, 100], stddev=.1))
        self.filter1 = tf.Variable(tf.random.normal([3, 3, 100, 100], stddev=.1))
        self.filter1 = tf.Variable(tf.random.normal([3, 3, 100, 20], stddev=.1))

        # now the value net specifics



        self.val1 = tf.keras.layers.Dense(
            100, activation='relu', trainable=True)
        self.val2 = tf.keras.layers.Dense(1, activation='tanh', trainable=True)
        
        # now the policy net specifics
        self.pol1 = tf.keras.layers.Dense(
            100, activation='relu', trainable=True)
        self.pol2 = tf.keras.layers.Dense(
            82, activation='softmax', trainable=True)

        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.mse = tf.keras.losses.MeanSquaredError()
        pass

    def callBase(self, game_batch):
        # game state is a list batch_size*games*gamesize(variable), for more info see PlayGames
        # we want to turn game_batch into a list of boards (with reversing perspective) and a list of captures
        boards = []
        playerCaps = []
        opponentCaps = []

        for game in game_batch:
            result = game[-1]
            for position in game[:-1]: # we throw away the result
                if position.to_play == 1:
                    boards.append(position.board)
                    playerCaps.append(position.caps[0])
                    opponentCaps.append(position.caps[1] + position.komi)
                else:
                    boards.append(position.board *-1)
                    playerCaps.append(position.caps[1] + position.komi)
                    opponentCaps.append(position.caps[0])

        boards = tf.convert_to_tensor(boards, dtype=tf.int32)
        playerCaps = tf.convert_to_tensor(playerCaps, dtype=tf.float32)
        opponentCaps = tf.convert_to_tensor(opponentCaps, dtype=tf.float32)
        this_size = len(boards)
        """
        this_size = len(game_state)
        batch = list(list(zip(*game_state)))
        board = tf.reshape(tf.convert_to_tensor(
            batch[0], dtype=tf.int32), [this_size, 9, 9, 1])
        
        myCaps = tf.convert_to_tensor(batch[1])

        oppCaps = tf.convert_to_tensor(batch[2])
        """
        boards = tf.reshape(boards, (this_size, 9, 9, 1))
        features = self.conv1(tf.cast(boards, tf.float32))
        features = self.norm1(features)
        features = self.conv2(features)
        features = self.norm2(features)
        features = self.conv3(features)
        features = self.norm3(features)

        features = tf.reshape(tf.cast(features, dtype=tf.float32), [this_size, 81, ])
        other_data = tf.convert_to_tensor([playerCaps, opponentCaps], dtype=tf.float32)
        other_data = tf.transpose(other_data)

        features = tf.concat([features, other_data], axis=1)
        """
        features = tf.nn.conv2d(boards, self.filter1, 1, padding='SAME')
        moments1 = tf.nn.moments(features, [0,1,2])
        features = tf.nn.batch_normalization(features, moments1[0], moments1[1], variance_epsilon=0.00001,offset= None, scale=None)
        features = tf.nn.conv2d(features, self.filter2, 1, padding='SAME')
        """

        return features
    
    def callVal(self, game_state):
        features = self.callBase(game_state)
        return self.val2(self.val1(features))

    def callPol(self, game_state):
        features = self.callBase(game_state)
        return self.pol2(self.pol1(features))



def playGames(model, number_of_games):
    games = []
    for _ in range(number_of_games):
        game = []
        position = go.Position()
        while not position.is_game_over():
            game.append(position)
            actions = model.callPol([[position, 1]])[0]
            actions = actions*position.all_legal_moves()
            move = tf.math.argmax(actions).numpy()
            print(move)
            print(position.__str__())
            position = position.play_move(coords.from_flat(move))
        game.append(position.result)
        games.append(game)
    return games




def loss(model, logits, labels):
    return model.mse(labels, logits)


def trainVal(model, data, labels):
    for start in range(0, len(data) - model.batch_size, model.batch_size):
        dataBatch = data[start:start+model.batch_size]
        labelBatch = labels[start:start+model.batch_size]
        with tf.GradientTape() as tape:
            losss = loss(model, model.callVal(dataBatch),
                         tf.convert_to_tensor(labelBatch))
        gradients = tape.gradient(losss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
    pass


def generateLabels(model, data):
    #for datum in data:
    pass


def trainPol(model, data):


    for start in range(0, len(data) - model.batch_size, model.batch_size):
        dataBatch = data[start:start+model.batch_size]
        # need to generate labels using MCTS



    # once we have the labels


white = player.Player()
black = player.Player()
data, labels = player.selfplay(black, white)


myModel = Model()
print(playGames(myModel, 1))
"""
trainVal(myVal, data, labels)

initialPos = go.Position()
mySearch = TreeSearch.MCTS(myVal, 30)
mySearch.run(myVal, initialPos).__repr__()


sess=tf.compat.v1.Session()
myVal=ValueNet()
saver=tf.compat.v1.train.Saver(myVal.trainable_variables, filename="val.txt")


# train
sess.run(train(myVal, data, labels))
saver.save(sess, "val.txt")

a = myVal.trainable_variables
print(a)
# print('aaaa')
# print(myVal.trainable_variables)
# print('aaaa')
sess = tf.compat.v1.Session()
# train
sess.run(a)
#saver.save(sess, "val.txt")
"""