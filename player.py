import go
import coords
import tensorflow as tf
import numpy as np
import time


class Player():
    def __init__(self):
        #nothing to do here yet
        pass

    def choose_and_play_move(self, position):
        legalMoves = position.all_legal_moves()
        pDistribution = tf.nn.softmax(tf.convert_to_tensor(legalMoves, dtype=tf.float64))
        choice = np.random.choice(np.arange(0, len(pDistribution)), p=pDistribution)
        while(legalMoves[choice] == 0):
            choice = np.random.choice(np.arange(0, len(pDistribution)), p=pDistribution)

        return position.play_move(coords.from_flat(choice))
"""
black = Player()
white = Player()
#we assume constant komi so there is no need to record this. 
allPositions = []
position = go.Position()
while not position.is_game_over():
    
    if (position.to_play == 1):
        position = black.choose_and_play_move(position)
    else:
        position = white.choose_and_play_move(position)
    print(position.__str__())
    print(position.board)
    time.sleep(1)
    allPositions.append((position.board, position.to_play*-1, position.caps))

print("The Score from Black's Perspective")
print(position.score())
"""

def selfplay(black, white):
    dataBlack = []
    dataWhite = []
    position = go.Position()
    while not position.is_game_over():
        if (position.to_play == 1):
            positionData = (np.array(position.board), position.caps[0], position.caps[1] + position.komi)
            dataBlack.append(positionData)
            # include komi?
            position = black.choose_and_play_move(position)
            
        else:
            positionData = (np.array(position.board)*-1, position.caps[1] + position.komi, position.caps[0])
            dataWhite.append(positionData)
            position = white.choose_and_play_move(position)
    data = []
    labels = []
    for datum in dataBlack:
        data.append(datum)
        labels.append(position.result())
    for datum in dataWhite:
        data.append(datum)
        labels.append(position.result()*-1)
        
    return data,labels
    

black = Player()
white = Player()        

# How should we represent the game state?
# we need the board, but do we want to alternate the 1s and -1s?
# we need captures

# The data from a self-play game would therefore look something like
# allPositions, where each element is labeled with a 1 or 0 depending on whether
# to_play*-1 (of each element) won the game. This is what we will use to train
# the value net. 
