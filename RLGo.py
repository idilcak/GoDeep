import go
import coords
import tensorflow as tf
import numpy as np
import time
import player
import math 













def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score

def all_ucb_score(parent):
    scores = np.zeros([82])
    scores += -1
    for childID in parent.children.keys():
        scores[childID] = ucb_score(parent, parent.children[childID])
    return scores


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
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
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
                self.children[a] = Node(prior=prob, to_play=self.position.to_play*-1)
            


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
                new_boards, new_playerCaps, new_opponentCaps = gamesToData([[next_position, 1]])
                action_probs = model.callPol(new_boards, new_playerCaps, new_opponentCaps)[0] 
                value = model.callVal(new_boards, new_playerCaps, new_opponentCaps)[0] 
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

    def callBase(self, all_boards, all_playerCaps, all_opponentCaps):
        # game state is a list batch_size*games*gamesize(variable), for more info see PlayGames
        # we want to turn game_batch into a list of boards (with reversing perspective) and a list of captures
        

        

        boards = tf.convert_to_tensor(all_boards, dtype=tf.int32)
        playerCaps = tf.convert_to_tensor(all_playerCaps, dtype=tf.float32)
        opponentCaps = tf.convert_to_tensor(all_opponentCaps, dtype=tf.float32)
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
    
    def callVal(self, boards, playerCaps, oppCaps):
        features = self.callBase(boards, playerCaps, oppCaps)
        return self.val2(self.val1(features))

    def callPol(self, boards, playerCaps, oppCaps):
        features = self.callBase(boards, playerCaps, oppCaps)
        return self.pol2(self.pol1(features))


# model: as defined in the class Model above
# number_of_games: pretty self explanatory
# returns a list of games: a game is a list of positions and with the result appended at the end
def playGames(model, number_of_games):
    games = []
    for _ in range(number_of_games):
        game = []
        position = go.Position()
        while not position.is_game_over():
            game.append(position)
            boards, playerCaps, opponentCaps = gamesToData([[position, 1]])
            actions = model.callPol(boards, playerCaps, opponentCaps)[0]
            actions = actions*position.all_legal_moves()
            move = tf.math.argmax(actions).numpy()
            position = position.play_move(coords.from_flat(move))
        game.append(position.result())
        games.append(game)
    return games


def gamesToData(games):
    boards = []
    playerCaps = []
    opponentCaps = []
    for game in games:
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
    return boards, playerCaps, opponentCaps



def gamesToResult(games):
    results = []
    for game in games:
        result = game[-1]
        for position in game[:-1]:
            if position.to_play == 1:
                results.append(result)
            else:
                results.append(result*-1)

    return results



def loss(model, logits, labels):
    return model.mse(labels, logits)







def trainVal(model, boards, playerCaps, opponentCaps, labels):
    #Shuffle Here
    for start in range(0, len(boards) - model.batch_size, model.batch_size):
        boardsBatch = boards[start:start+model.batch_size]
        playerCapsBatch = playerCaps[start:start+model.batch_size]
        opponentCapsBatch = opponentCaps[start:start+model.batch_size]
        labelBatch = labels[start:start+model.batch_size]
        with tf.GradientTape() as tape:
            losss = loss(model, model.callVal(boardsBatch, playerCapsBatch, opponentCapsBatch),
                         tf.convert_to_tensor(labelBatch))
        gradients = tape.gradient(losss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

    pass

def generatePolLabels(model, positions):
    myTree = MCTS(model, 300)
    labels = []
    for position in positions:
        labels.append(all_ucb_score(myTree.run(model, position)))
    return labels

def trainPol(model, games, boards, playerCaps, opponentCaps):
    positions = []
    for game in games:
        for position in game[:-1]:
            positions.append(position)
    # shuffle here
    for start in range(0, len(boards) - model.batch_size, model.batch_size):
        boardsBatch = boards[start:start+model.batch_size]
        playerCapsBatch = playerCaps[start:start+model.batch_size]
        opponentCapsBatch = opponentCaps[start:start+model.batch_size]
        positionBatch = positions[start:start+model.batch_size]
        labelsBatch = generatePolLabels(model, positionBatch)
        with tf.GradientTape() as tape:
            losss = loss(model, model.callPol(boardsBatch, playerCapsBatch, opponentCapsBatch), labelsBatch)
        gradients = tape.gradient(losss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    pass


def demonstration(model):
    position = go.Position()
    while not position.is_game_over():
        boards, playerCaps, opponentCaps = gamesToData([[position, 1]])
        actions = model.callPol(boards, playerCaps, opponentCaps)[0]
        actions = actions*position.all_legal_moves()
        move = tf.math.argmax(actions).numpy()
        print(position.__str__())
        position = position.play_move(coords.from_flat(move))
    pass




myModel = Model()



for _ in range(10):
    games = playGames(myModel, 5)
    print(games[0])
    boards, playerCaps, opponentCaps = gamesToData(games)
    labels = gamesToResult(games)
    trainVal(myModel, boards, playerCaps, opponentCaps, labels)
    trainPol(myModel, games, boards, playerCaps, opponentCaps)
demonstration(myModel)

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