import go
import coords
import tensorflow as tf
import numpy as np
import time
import math




# model: as defined in the class Model above
# number_of_games: pretty self explanatory
# returns a list of games: a game is a list of positions and with the result appended at the end
def playGames(model, number_of_games):
    # will restrict the games to a maximum of 102 moves
    games = []
    for _ in range(number_of_games):
        game = []
        position = go.Position()
        while not position.is_game_over():
            if position.n >= 100:
                position = position.pass_move()
                game.append(position)

            else:
                game.append(position)
                boards, playerCaps, opponentCaps = gamesToData([[position, 1]])
                actions = model.callPol(boards, playerCaps, opponentCaps)[0]
                pdist = tf.nn.softmax(tf.cast(actions, dtype=tf.float64))
                legalMoves = position.all_legal_moves()
                move = np.random.choice(np.arange(0, len(pdist)), p=pdist)
                if legalMoves[move] == 0:
                    actions = actions*legalMoves
                    move = tf.math.argmax(actions).numpy()
                position = position.play_move(coords.from_flat(move))
        game.append(position.result())
        games.append(game)
    return games


#takes a list of lists of positions of varying length and returns data in a useful format
#return, a tuple with boards, playerCaps and opponentCaps
# boards is a list of varying size whose elements are 2d lists of size (9,9) and represent a board
# playerCaps is a list of varying size (same size as all_boards) whose elements are floats representing
# these boards are perspective invariant.
# the stones by the player whose turn it is to place in the baord of the same index
# opponetCaps is a list of varying size (same size as all_boards) whose elements are floats representing
# the stones by the opponent of the player whose turn it is to place in the baord of the same index
def gamesToData(games):
    boards = []
    playerCaps = []
    opponentCaps = []
    for game in games:
        for position in game[:-1]:  # we throw away the result which is the last element of every game
            if position.to_play == 1:
                boards.append(position.board)
                playerCaps.append(position.caps[0])
                opponentCaps.append(position.caps[1] + position.komi)
            else:
                boards.append(position.board * -1)
                playerCaps.append(position.caps[1] + position.komi)
                opponentCaps.append(position.caps[0])
    return boards, playerCaps, opponentCaps


# takes a list of lists of positions encoding games and returns a flattened list of results (
# -1 if the player to play in the same index of position lost, 1 if they won) Furthermore it will discount
# earlier positions in the game to encourage the model to keep playing (for a detailed discussion of why we did this
# refer to the write up)
def gamesToResult(games):
    results = []
    for game in games:
        result = game[-1]
        for i in range(len(game)-1):
            if game[i].to_play == 1:
                results.append(result*(0.98**(100-i)))
            else:
                results.append(result*-1*(0.98**(100-i)))

    return results

