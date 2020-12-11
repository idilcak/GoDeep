# GoDeep
The files:
- the go.py and the coords.py files are taken from Minigo and contain functions related to the environment. 
- preprocessing.py has three functions: playGames, gamesToData, and gamesToResult
    - playGames takes a model and does the self-play stage, returning a list of games, where a game is a list of positions with the result appended at the end
    - gamesToData and gamesToResult are used to extract the information that is required from positions for training
- RLGo.py has the MCTS classes and methods as well as the training
    - ucb_score, Node class, MCTS class were all written by Josh Varty
    - all_ucb_score and generatePolLabels were written for us, and they are used to run the MCTS and get the probabilities
    - class Model has the model definition, both value net and policy net is within this class
        - callBase is a function that has the layers shared by the value and policy network, callVal is for the value network specifics, and callPol is for the policy network specifics
    - the loss functions and the train functions are here
    - at the end of the RLGo.py file, we have the training loop to train the model
- testingFuncs.py has all the methods that have to do with sparring to test how well the model is doing



Instructions:
    The main file from which to control everything is the RLGo.py file. In here, at the bottom we have set up a train and testing loop. If you want to train and test for 50 games (this will probably take more than an hour but there will be things to see along the way), leave the file as it is and run it. Every 3 games the training model will play against a random player 40 times. And after the fifth game, every 4 games it will play against the version of itself trained at 5 games.
    If you want to change the number of games run change the for loop's endpoint.
    If you want to spar more or less often change the conditionals at the beginning of the loop.
    If you want to see a game played by the model include the line demonstration(myModel)

Note: Ignore the TensorFlow warning about gradients not existing. Tensorflow gives this warning because when training the Value Network there are no gradients for the Policy Network and vice versa.