import go
import coords
import tensorflow as tf
import numpy as np
import time
import math
from preprocessing import *


# takes a model and makes it play against itself, then it shows every move played
def demonstration(model):
    position = go.Position()
    while not position.is_game_over():
        if position.n >= 100:
            position = position.pass_move()
        else:
            boards, playerCaps, opponentCaps = gamesToData([[position, 1]])
            actions = model.callPol(boards, playerCaps, opponentCaps)[0]
            actions = actions*position.all_legal_moves()
            move = tf.math.argmax(actions).numpy()
            print(position.__str__())
            position = position.play_move(coords.from_flat(move))
    pass


# takes two models veteran and beginner and makes them play against each other for as many matches as specified
# by the parameter 'matches'
def spar(veteran, beginner, matches):
    veteranWins = 0
    beginnerWins = 0
    for i in range(matches):
        if i % 2 == 0:
            black = veteran
            white = beginner
        else:
            black = beginner
            white = veteran
        position = go.Position()
        while not position.is_game_over():
            if position.n>=100:
                position = position.pass_move()
            else:
                if position.to_play == 1:
                    boards, playerCaps, opponentCaps = gamesToData([[position, 1]])
                    actions = black.callPol(boards, playerCaps, opponentCaps)[0]
                    pdist = tf.nn.softmax(tf.cast(actions, dtype=tf.float64))
                    legalMoves = position.all_legal_moves()
                    move = np.random.choice(np.arange(0, len(pdist)), p=pdist)
                    if legalMoves[move] == 0:
                        actions = actions*legalMoves
                        move = tf.math.argmax(actions).numpy()
                    position = position.play_move(coords.from_flat(move))
                else:
                    boards, playerCaps, opponentCaps = gamesToData([[position, 1]])
                    actions = white.callPol(boards, playerCaps, opponentCaps)[0]
                    pdist = tf.nn.softmax(tf.cast(actions, dtype=tf.float64))
                    legalMoves = position.all_legal_moves()
                    move = np.random.choice(np.arange(0, len(pdist)), p=pdist)
                    if legalMoves[move] == 0:
                        actions = actions*legalMoves
                        move = tf.math.argmax(actions).numpy()
                    position = position.play_move(coords.from_flat(move))
        if black == veteran:
            if position.result() == 1:
                veteranWins += 1
            elif position.result() == -1:
                beginnerWins += 1
            else:
                print("No one wins!!")
        else:
            if position.result() == 1:
                beginnerWins += 1
            elif position.result() == -1:
                veteranWins += 1
            else:
                print("No one wins!!")
    print("The veteran wins "+str(veteranWins))
    print("The beginner wins "+str(beginnerWins))
    return veteranWins - beginnerWins


def choose_and_play_move(position):
    # position: a go.py position object
    # plays a random legal move from the position
    legalMoves = position.all_legal_moves()
    pDistribution = tf.nn.softmax(tf.convert_to_tensor(legalMoves, dtype=tf.float64))
    choice = np.random.choice(np.arange(0, len(pDistribution)), p=pDistribution)
    while(legalMoves[choice] == 0):
        choice = np.random.choice(np.arange(0, len(pDistribution)), p=pDistribution)

    return position.play_move(coords.from_flat(choice))


def testAgainstRandom(model, matches):
    #untrained models do not play at random, they have random weight initializations and then alawys play in terms of those
    #this function takes a model (a trained one) and plays it against a player who makes a random move every time.
    #it plays matches number of matches
    veteran = model
    veteranWins = 0
    beginnerWins = 0
    white = None
    black = None
    for i in range(matches):
        if i % 2 == 0:
            black = veteran
        else:
            white = veteran
        position = go.Position()
        while not position.is_game_over():
            if position.n>=100:
                    position = position.pass_move()
            else:
                if position.to_play == 1:
                    if black == veteran:
                        boards, playerCaps, opponentCaps = gamesToData([[position, 1]])
                        actions = black.callPol(boards, playerCaps, opponentCaps)[0]
                        pdist = tf.nn.softmax(tf.cast(actions, dtype=tf.float64))
                        legalMoves = position.all_legal_moves()
                        move = np.random.choice(np.arange(0, len(pdist)), p=pdist)
                        if legalMoves[move] == 0:
                            actions = actions*legalMoves
                            move = tf.math.argmax(actions).numpy()
                        position = position.play_move(coords.from_flat(move))
                    else:
                        position = choose_and_play_move(position)
                else:
                    if white == veteran:
                        boards, playerCaps, opponentCaps = gamesToData([[position, 1]])
                        actions = white.callPol(boards, playerCaps, opponentCaps)[0]
                        pdist = tf.nn.softmax(tf.cast(actions, dtype=tf.float64))
                        legalMoves = position.all_legal_moves()
                        move = np.random.choice(np.arange(0, len(pdist)), p=pdist)
                        if legalMoves[move] == 0:
                            actions = actions*legalMoves
                            move = tf.math.argmax(actions).numpy()
                        position = position.play_move(coords.from_flat(move))
                    else:
                        position = choose_and_play_move(position)
                
        if black == veteran:
            if position.result() == 1:
                veteranWins += 1
            elif position.result() == -1:
                beginnerWins += 1
            else:
                print("No one wins!!")
        else:
            if position.result() == 1:
                beginnerWins += 1
            elif position.result() == -1:
                veteranWins += 1
            else:
                print("No one wins!!")
    print("The model wins "+str(veteranWins))
    print("The random wins "+str(beginnerWins))
    return veteranWins - beginnerWins