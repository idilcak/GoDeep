import go
import coords
import tensorflow as tf
import numpy as np
import time
import math
from preprocessing import *
from testingFuncs import *


##MONTE CARLO TREE SEARCH CLASSES AND METHODS WRITTEN BY JOSH VARTY, modified by us##
def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * \
        math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node():
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.position = None
        self.to_play = to_play

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array(
            [child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / \
                sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, position, action_probs):

        self.position = position
        for a, prob in enumerate(action_probs):

            if prob != 0:
                self.children[a] = Node(
                    prior=prob, to_play=self.position.to_play*-1)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.position.__str__(), prior, self.visit_count, self.value())






class MCTS():

    def __init__(self, model, number_of_sim):
        self.model = model
        self.number_of_sim = number_of_sim

    def run(self, model, position):
        # assert position is of type Position from go.py
        root = Node(0, position.to_play)
        boards, playerCaps, opponentCaps = gamesToData([[position, 1]])
        action_probs = model.callPol(boards, playerCaps, opponentCaps)[0]
        value = model.callVal(boards, playerCaps, opponentCaps)[0]
        valid_moves = position.all_legal_moves()
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)
        root.expand(position, action_probs)

        for _ in range(self.number_of_sim):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)
            parent = search_path[-2]
            position = parent.position
            next_position = position.play_move(coords.from_flat(action))
            if not next_position.is_game_over():
                new_boards, new_playerCaps, new_opponentCaps = gamesToData(
                    [[next_position, 1]])
                action_probs = model.callPol(
                    new_boards, new_playerCaps, new_opponentCaps)[0]
                value = model.callVal(
                    new_boards, new_playerCaps, new_opponentCaps)[0]
                valid_moves = next_position.all_legal_moves()
                action_probs = action_probs * valid_moves  # mask invalid moves
                action_probs /= np.sum(action_probs)

                node.expand(next_position, action_probs)
            else:
                if next_position.to_play == 1:
                    value = next_position.result()
                else:
                    value = next_position.result()*-1

            self.backpropagate(search_path, value, next_position.to_play)
        return root

    def backpropagate(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1


## MCTS methods written only by us ##

def all_ucb_score(parent):
    scores = np.zeros([82])
    scores += -1
    for childID in parent.children.keys():
        scores[childID] = ucb_score(parent, parent.children[childID])*-1
    return scores


def generatePolLabels(model, positions):
    myTree = MCTS(model, 65)
    labels = []
    for position in positions:
        labels.append(all_ucb_score(myTree.run(model, position)))
    return labels








## MODEL DEFINITION


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 20

        #shared layers
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

        # now the policy net specifics
        self.pol1 = tf.keras.layers.Dense(
            100, activation='relu', trainable=True)
        self.pol2 = tf.keras.layers.Dense(
            82, activation='softmax', trainable=True)
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.mse = tf.keras.losses.MeanSquaredError()
        pass
    
    #method that calls the layers shared by the policy and value Networks
    def callBase(self, all_boards, all_playerCaps, all_opponentCaps):
        # all_boards is a list of varying size whose elements are 2d lists of size (9,9) and represent a board
        # all_playerCaps is a list of varying size (same size as all_boards) whose elements are floats representing
        # the stones by the player whose turn it is to place in the baord of the same index
        # all_opponetCaps is a list of varying size (same size as all_boards) whose elements are floats representing
        # the stones by the opponent of the player whose turn it is to place in the baord of the same index

        boards = tf.convert_to_tensor(all_boards, dtype=tf.int32)
        playerCaps = tf.convert_to_tensor(all_playerCaps, dtype=tf.float32)
        opponentCaps = tf.convert_to_tensor(all_opponentCaps, dtype=tf.float32)
        this_size = len(boards)
        boards = tf.reshape(boards, (this_size, 9, 9, 1))
        features = self.conv1(tf.cast(boards, tf.float32))
        features = self.norm1(features)
        features = self.conv2(features)
        features = self.norm2(features)
        features = self.conv3(features)
        features = self.norm3(features)

        features = tf.reshape(
            tf.cast(features, dtype=tf.float32), [this_size, 81, ])
        other_data = tf.convert_to_tensor(
            [playerCaps, opponentCaps], dtype=tf.float32)
        other_data = tf.transpose(other_data)

        features = tf.concat([features, other_data], axis=1)
        return features

    def callVal(self, boards, playerCaps, oppCaps):
        #inputs are the same as in callBase, calls Base and then applies the Dense layers for the Value Network
        features = self.callBase(boards, playerCaps, oppCaps)
        return self.val2(self.val1(features))

    def callPol(self, boards, playerCaps, oppCaps):
        #inputs are the same as in callBase, calls Base and then applies the Dense layers for the Policy Network

        features = self.callBase(boards, playerCaps, oppCaps)
        return self.pol2(self.pol1(features))




# calculates mean squared error loss for model based on labels and logits (these can be either value network 
# loss or policy network loss)

def loss(model, logits, labels):
    return model.mse(labels, logits)


#training loop for Value network
# boards is a list of varying size whose elements are 2d lists of size (9,9) and represent a board
# playerCaps is a list of varying size (same size as all_boards) whose elements are floats representing
# the stones by the player whose turn it is to place in the baord of the same index
# opponetCaps is a list of varying size (same size as all_boards) whose elements are floats representing
# the stones by the opponent of the player whose turn it is to place in the baord of the same index
# labels: the discounted results of each position, see preprocessing function position to result
def trainVal(model, boards, playerCaps, opponentCaps, labels):
    # Shuffle Here
    ind = range(len(boards))
    ind = tf.random.shuffle(ind)
    boards = tf.gather(boards, ind)
    playerCaps = tf.gather(playerCaps, ind)
    opponentCaps = tf.gather(opponentCaps, ind)
    labels = tf.gather(labels, ind)
    for start in range(0, len(boards) - model.batch_size, model.batch_size):
        #batch the relevant data 
        boardsBatch = boards[start:start+model.batch_size]
        playerCapsBatch = playerCaps[start:start+model.batch_size]
        opponentCapsBatch = opponentCaps[start:start+model.batch_size]
        labelBatch = labels[start:start+model.batch_size]
        #calculate loss
        with tf.GradientTape() as tape:
            losss = loss(model, model.callVal(boardsBatch, playerCapsBatch, opponentCapsBatch),
                         tf.convert_to_tensor(labelBatch))
        gradients = tape.gradient(losss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

    pass


#training loop for Value network
# boards is a list of varying size whose elements are 2d lists of size (9,9) and represent a board
# playerCaps is a list of varying size (same size as all_boards) whose elements are floats representing
# the stones by the player whose turn it is to place in the baord of the same index
# opponetCaps is a list of varying size (same size as all_boards) whose elements are floats representing
# the stones by the opponent of the player whose turn it is to place in the baord of the same index
# games is a list of lists of positions as produced by the playGames function in preprocessing
def trainPol(model, games, boards, playerCaps, opponentCaps):
    positions = []
    #flatten the positions
    for game in games:
        for position in game[:-1]:
            positions.append(position)
    for start in range(0, len(boards) - model.batch_size, model.batch_size):
        boardsBatch = boards[start:start+model.batch_size]
        playerCapsBatch = playerCaps[start:start+model.batch_size]
        opponentCapsBatch = opponentCaps[start:start+model.batch_size]
        positionBatch = positions[start:start+model.batch_size]
        #use the generatePolLabels function to run the MCTS
        labelsBatch = generatePolLabels(model, positionBatch)
        with tf.GradientTape() as tape:
            losss = loss(model, model.callPol(
                boardsBatch, playerCapsBatch, opponentCapsBatch), labelsBatch)
        gradients = tape.gradient(losss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
    pass





## TRAINING LOOP ##

white_wins = 0
black_wins = 0
games_length = []
myModel = Model()

newModel = Model()

for i in range(50):
    ##SOME RELEVANT TESTING
    #every three games it spars against the random player
    if i % 3 == 2:
        print("Spar against random player")
        testAgainstRandom(myModel, 40)
    if i == 5:
        novice = myModel
    #after the 5th move and every 4 moves it plays against the version of itself that was trained on 5 games
    if i % 4 == 1 and i > 5:
        print("Spar against novice")
        spar(myModel, novice, 40)
    print(str(i+1) + " out of 100")
    games = playGames(myModel, 1)
    #want to save the number of games white won, the number black won and the number of moves per game
    for game in games:
        print("The game's length was " +str(len(game)-1))
        games_length.append(len(game)-1)
        result = game[-1]
        if result == 1:
            black_wins += 1
        elif result == -1:
            white_wins += 1
        else:
            print("No one wins!!") #this is bad, there are no draws in Go
    #collect data
    boards, playerCaps, opponentCaps = gamesToData(games)
    #create labels for value network
    labels = gamesToResult(games)
    #train value network
    trainVal(myModel, boards, playerCaps, opponentCaps, labels)
    #train policy network
    trainPol(myModel, games, boards, playerCaps, opponentCaps)
    print("done with training")


print("The number of black wins " + str(black_wins))
print("The number of white wins " + str(white_wins))
print("The game lengths")
print(games_length)

print("Sparring against untrained version of self")
spar(myModel, newModel, 200)
print("Sparring against version after 5 games")
spar(myModel, novice, 200)
print("Sparring against random player")
testAgainstRandom(myModel, 200)
myModel.save_weights('my-weights.txt')
