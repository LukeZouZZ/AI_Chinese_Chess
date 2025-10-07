"""Monte Carlo Tree Search (MCTS) implementation"""

import numpy as np
import copy
from config import CONFIG


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# Define a tree node
class TreeNode(object):
    """
    Node in the MCTS tree.
    Tracks Q (mean value), P (prior), and U (confidence bound).
    """

    def __init__(self, parent, prior_p):
        """
        Args:
            parent: parent node
            prior_p: prior probability of selecting this move
        """
        self._parent = parent
        self._children = {}  # mapping: action -> TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand the tree by creating new children."""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select the child node maximizing Q+U."""
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """Compute and return node value: Q + U."""
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        """Update node value from leaf evaluation."""
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Recursively update ancestors and self."""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """Whether this node has no children."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """Monte Carlo Tree Search engine."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        """policy_value_fn(board) -> ([(action, prob)], value)."""
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run one simulation and backpropagate results."""
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf node
        action_probs, leaf_value = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player_id() else -1.0
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run simulations and return move probabilities."""
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(a, n._n_visits) for a, n in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """Move the root to the selected child."""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """MCTS-based AI player."""

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = "AI"

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        """Reset search tree."""
        self.mcts.update_with_move(-1)

    def __str__(self):
        return "MCTS {}".format(self.player)

    def get_action(self, board, temp=1e-3, return_prob=0):
        """Return the chosen move (and optional move probability vector)."""
        move_probs = np.zeros(2086)
        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            move = np.random.choice(
                acts,
                p=0.75 * probs + 0.25 * np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs)))
            )
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
        if return_prob:
            return move, move_probs
        else:
            return move
