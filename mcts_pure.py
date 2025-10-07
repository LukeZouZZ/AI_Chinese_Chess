# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)

@author: Junxiao Song
"""

import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(board):
    """A coarse and fast policy function used in the rollout phase.

    It generates random probabilities for available actions to drive a quick rollout.
    """
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """A function that takes in a state and outputs a list of (action, probability)
    tuples and a scalar score for the state.

    Here, for pure MCTS, we return uniform probabilities and zero score.
    """
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    """A node in the MCTS tree.
    Each node keeps track of its value Q, prior probability P, and visit-adjusted prior u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # map: action -> TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand the tree by creating new children from (action, prior) pairs."""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select the child action that maximizes Q + u(P).

        Returns:
            (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update this node's value statistics from a leaf evaluation.

        Args:
            leaf_value: value of the subtree from the current player's perspective.
        """
        # Count visit
        self._n_visits += 1
        # Running average for Q
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Recursively update all ancestors (this node last)."""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Compute the node's score Q + u.

        u balances prior P with visit count to encourage exploration.

        Args:
            c_puct: controls relative impact of Q vs P.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Whether this node has no expanded children."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        Args:
            policy_value_fn: callable(state) -> (iterable[(action, prob)], value in [-1,1])
            c_puct: higher values rely more on prior to explore
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from root to leaf, evaluate leaf, and backpropagate.

        Note:
            `state` is modified in-place; pass a copy when calling this.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        # Check terminal
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        # Rollout to get leaf value
        leaf_value = self._evaluate_rollout(state)
        # Backpropagate
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """Roll out until game end.

        Returns:
            +1 if current player wins, -1 if loses, 0 for tie.
        """
        player = state.get_current_player_id()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # No break: reached rollout limit
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        """Run all playouts and return the most visited action."""
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Advance the root to the given move if it exists; otherwise reset."""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTS_Pure(object):
    """AI player based on pure MCTS (no neural network)."""
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
