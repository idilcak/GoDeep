import math
import numpy as np
import go
import coords
import RLGo


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
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())




class MCTS():
    
    def __init__(self, model, number_of_sim):
        self.model = model
        self.number_of_sim = number_of_sim
    

    def run(self, model, position):
        # assert position is of type Position from go.py
        root = Node(0, position.to_play)
        boards, playerCaps, opponentCaps = RLGo.gamesToData([[position, 1]])
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
                new_boards, new_playerCaps, new_opponentCaps = RLGo.gamesToData([[position, 1]])
                action_probs = model.callPol(new_boards, new_playerCaps, new_opponentCaps)[0] 
                value = model.callVal(new_boards, new_playerCaps, new_opponentCaps)[0] 
                valid_moves = position.all_legal_moves()
                action_probs = action_probs * valid_moves  # mask invalid moves
                action_probs /= np.sum(action_probs)
                root.expand(next_position, action_probs)
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



def newTree(model, number_of_sim):
    return MCTS(model, number_of_sim)


