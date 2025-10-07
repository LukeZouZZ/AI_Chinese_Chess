"""Board game control (Xiangqi / Chinese Chess)"""

import numpy as np
import copy
import time
from config import CONFIG
from collections import deque   # used to detect perpetual check/chase (repetition)
import random


# Represent the board as a list of lists; Red on top, Black on bottom.
# Use deepcopy when operating on it.
state_list_init = [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
                   ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
                   ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']]


# Use a deque to store recent board states (length 4) for repetition checks
state_deque_init = deque(maxlen=4)
for _ in range(4):
    state_deque_init.append(copy.deepcopy(state_list_init))


# Build a mapping: piece-string -> one-hot array; and a reverse function array -> string
string2array = dict(红车=np.array([1, 0, 0, 0, 0, 0, 0]), 红马=np.array([0, 1, 0, 0, 0, 0, 0]),
                    红象=np.array([0, 0, 1, 0, 0, 0, 0]), 红士=np.array([0, 0, 0, 1, 0, 0, 0]),
                    红帅=np.array([0, 0, 0, 0, 1, 0, 0]), 红炮=np.array([0, 0, 0, 0, 0, 1, 0]),
                    红兵=np.array([0, 0, 0, 0, 0, 0, 1]), 黑车=np.array([-1, 0, 0, 0, 0, 0, 0]),
                    黑马=np.array([0, -1, 0, 0, 0, 0, 0]), 黑象=np.array([0, 0, -1, 0, 0, 0, 0]),
                    黑士=np.array([0, 0, 0, -1, 0, 0, 0]), 黑帅=np.array([0, 0, 0, 0, -1, 0, 0]),
                    黑炮=np.array([0, 0, 0, 0, 0, -1, 0]), 黑兵=np.array([0, 0, 0, 0, 0, 0, -1]),
                    一一=np.array([0, 0, 0, 0, 0, 0, 0]))


def array2string(array):
    return list(filter(lambda string: (string2array[string] == array).all(), string2array))[0]


# Apply a move to produce a new board state
def change_state(state_list, move):
    """move : string like '0010' (from_y, from_x, to_y, to_x)"""
    copy_list = copy.deepcopy(state_list)
    y, x, toy, tox = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    copy_list[toy][tox] = copy_list[y][x]
    copy_list[y][x] = '一一'
    return copy_list


# Pretty-print the board (for visualization)
def print_board(_state_array):
    # _state_array: [10, 9, 7], HWC
    board_line = []
    for i in range(10):
        for j in range(9):
            board_line.append(array2string(_state_array[i][j]))
        print(board_line)
        board_line.clear()


# Convert list-based board state to array-based state
def state_list2state_array(state_list):
    _state_array = np.zeros([10, 9, 7])
    for i in range(10):
        for j in range(9):
            _state_array[i][j] = string2array[state_list[i][j]]
    return _state_array


# Construct the complete set of legal move labels (length 2086),
# which is also the length of the NN move-probability vector.
# First dict: move_id -> move_action
# Second dict: move_action -> move_id
# Example: move_id:0 --> move_action:'0010'
def get_all_legal_moves():
    _move_id2move_action = {}
    _move_action2move_id = {}
    row = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # All advisor (guard) moves within the palace
    advisor_labels = ['0314', '1403', '0514', '1405', '2314', '1423', '2514', '1425',
                      '9384', '8493', '9584', '8495', '7384', '8473', '7584', '8475']
    # All bishop (elephant) diagonal moves
    bishop_labels = ['2002', '0220', '2042', '4220', '0224', '2402', '4224', '2442',
                     '2406', '0624', '2446', '4624', '0628', '2806', '4628', '2846',
                     '7052', '5270', '7092', '9270', '5274', '7452', '9274', '7492',
                     '7456', '5674', '7496', '9674', '5678', '7856', '9678', '7896']
    idx = 0
    for l1 in range(10):
        for n1 in range(9):
            # Combine rook-like (rank/file) and knight-like destinations
            destinations = [(t, n1) for t in range(10)] + \
                           [(l1, t) for t in range(9)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]  # knight L-shape
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(10) and n2 in range(9):
                    action = column[l1] + row[n1] + column[l2] + row[n2]
                    _move_id2move_action[idx] = action
                    _move_action2move_id[action] = idx
                    idx += 1

    for action in advisor_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1

    for action in bishop_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1

    return _move_id2move_action, _move_action2move_id


move_id2move_action, move_action2move_id = get_all_legal_moves()


# Move flip function used for data augmentation
def flip_map(string):
    new_str = ''
    for index in range(4):
        if index == 0 or index == 2:
            new_str += (str(string[index]))
        else:
            new_str += (str(8 - int(string[index])))
    return new_str


# Boundary check for board coordinates
def check_bounds(toY, toX):
    if toY in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] and toX in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        return True
    return False


# Cannot move to a square occupied by your own piece
def check_obstruct(piece, current_player_color):
    # If the destination contains a piece, check ownership
    if piece != '一一':
        if current_player_color == '红':
            if '黑' in piece:
                return True
            else:
                return False
        elif current_player_color == '黑':
            if '红' in piece:
                return True
            else:
                return False
    else:
        return True


# Get the set of legal moves for the current board
# Input state_deque must have length >= 10; current_player_color is the side to move
# Return a list of legal move IDs, e.g., [0, 1, 2, 1089, 2085]
def get_legal_moves(state_deque, current_player_color):
    """
    Repetition avoidance example (rook chasing the general/king):

    ====
      将
    车
    ====
    ====
      将
      车
    ====
    ====
    将
      车
    ====
    ====
    将
    车
    ====
    ====
      将
    车
    ====

    In this situation, the rook cannot continue moving right to chase the general.
    The next disallowed action would be (1011), because the resulting board
    would repeat the position at state_deque[-4].
    """

    state_list = state_deque[-1]
    old_state_list = state_deque[-4]

    moves = []  # store all legal moves as strings
    face_to_face = False  # whether the two generals face each other directly

    # Record positions of the two generals
    k_x = None
    k_y = None
    K_x = None
    K_y = None

    # state_list is a list-of-lists board: len(state_list) == 10, len(state_list[0]) == 9
    # Iterate over all origin squares
    for y in range(10):
        for x in range(9):
            # Only squares with a piece can move
            if state_list[y][x] == '一一':
                pass
            else:
                if state_list[y][x] == '黑车' and current_player_color == '黑':  # legal moves for black rook
                    toY = y
                    for toX in range(x - 1, -1, -1):
                        # m encodes from->to; rook cannot jump over pieces (break on first block)
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '红' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    for toX in range(x + 1, 9):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '红' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)

                    toX = x
                    for toY in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '红' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    for toY in range(y + 1, 10):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '红' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)

                elif state_list[y][x] == '红车' and current_player_color == '红':  # legal moves for red rook
                    toY = y
                    for toX in range(x - 1, -1, -1):
                        # rook cannot jump over pieces (break on first block)
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '黑' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    for toX in range(x + 1, 9):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '黑' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)

                    toX = x
                    for toY in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '黑' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    for toY in range(y + 1, 10):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '黑' in state_list[toY][toX]:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)

                # Legal moves for black knight
                elif state_list[y][x] == '黑马' and current_player_color == '黑':
                    for i in range(-1, 3, 2):
                        for j in range(-1, 3, 2):
                            toY = y + 2 * i
                            toX = x + 1 * j
                            if check_bounds(toY, toX) \
                                    and check_obstruct(state_list[toY][toX], current_player_color='黑') \
                                    and state_list[toY - i][x] == '一一':
                                m = str(y) + str(x) + str(toY) + str(toX)
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            toY = y + 1 * i
                            toX = x + 2 * j
                            if check_bounds(toY, toX) \
                                    and check_obstruct(state_list[toY][toX], current_player_color='黑') \
                                    and state_list[y][toX - j] == '一一':
                                m = str(y) + str(x) + str(toY) + str(toX)
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)

                # Legal moves for red knight
                elif state_list[y][x] == '红马' and current_player_color == '红':
                    for i in range(-1, 3, 2):
                        for j in range(-1, 3, 2):
                            toY = y + 2 * i
                            toX = x + 1 * j
                            if check_bounds(toY, toX) \
                                    and check_obstruct(state_list[toY][toX], current_player_color='红') \
                                    and state_list[toY - i][x] == '一一':
                                m = str(y) + str(x) + str(toY) + str(toX)
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            toY = y + 1 * i
                            toX = x + 2 * j
                            if check_bounds(toY, toX) \
                                    and check_obstruct(state_list[toY][toX], current_player_color='红') \
                                    and state_list[y][toX - j] == '一一':
                                m = str(y) + str(x) + str(toY) + str(toX)
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)

                # Legal moves for black bishop (elephant)
                elif state_list[y][x] == '黑象' and current_player_color == '黑':
                    for i in range(-2, 3, 4):
                        toY = y + i
                        toX = x + i
                        if check_bounds(toY, toX) \
                                and check_obstruct(state_list[toY][toX], current_player_color='黑') \
                                and toY >= 5 and state_list[y + i // 2][x + i // 2] == '一一':
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        toY = y + i
                        toX = x - i
                        if check_bounds(toY, toX) \
                                and check_obstruct(state_list[toY][toX], current_player_color='黑') \
                                and toY >= 5 and state_list[y + i // 2][x - i // 2] == '一一':
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                # Legal moves for red bishop (elephant)
                elif state_list[y][x] == '红象' and current_player_color == '红':
                    for i in range(-2, 3, 4):
                        toY = y + i
                        toX = x + i
                        if check_bounds(toY, toX) \
                                and check_obstruct(state_list[toY][toX], current_player_color='红') \
                                and toY <= 4 and state_list[y + i // 2][x + i // 2] == '一一':
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        toY = y + i
                        toX = x - i
                        if check_bounds(toY, toX) \
                                and check_obstruct(state_list[toY][toX], current_player_color='红') \
                                and toY <= 4 and state_list[y + i // 2][x - i // 2] == '一一':
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                # Legal moves for black advisor (guard)
                elif state_list[y][x] == '黑士' and current_player_color == '黑':
                    for i in range(-1, 3, 2):
                        toY = y + i
                        toX = x + i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='黑') \
                                and toY >= 7 and 3 <= toX <= 5:
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        toY = y + i
                        toX = x - i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='黑') \
                                and toY >= 7 and 3 <= toX <= 5:
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                # Legal moves for red advisor (guard)
                elif state_list[y][x] == '红士' and current_player_color == '红':
                    for i in range(-1, 3, 2):
                        toY = y + i
                        toX = x + i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='红') \
                                and toY <= 2 and 3 <= toX <= 5:
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        toY = y + i
                        toX = x - i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='红') \
                                and toY <= 2 and 3 <= toX <= 5:
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                # Legal moves for black general (king)
                elif state_list[y][x] == '黑帅':
                    k_x = x
                    k_y = y
                    if current_player_color == '黑':
                        for i in range(2):
                            for sign in range(-1, 2, 2):
                                j = 1 - i
                                toY = y + i * sign
                                toX = x + j * sign

                                if check_bounds(toY, toX) and check_obstruct(
                                        state_list[toY][toX], current_player_color='黑') and toY >= 7 and 3 <= toX <= 5:
                                    m = str(y) + str(x) + str(toY) + str(toX)
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)

                # Legal moves for red general (king)
                elif state_list[y][x] == '红帅':
                    K_x = x
                    K_y = y
                    if current_player_color == '红':
                        for i in range(2):
                            for sign in range(-1, 2, 2):
                                j = 1 - i
                                toY = y + i * sign
                                toX = x + j * sign

                                if check_bounds(toY, toX) and check_obstruct(
                                        state_list[toY][toX], current_player_color='红') and toY <= 2 and 3 <= toX <= 5:
                                    m = str(y) + str(x) + str(toY) + str(toX)
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)

                # Legal moves for black cannon
                elif state_list[y][x] == '黑炮' and current_player_color == '黑':
                    toY = y
                    hits = False
                    for toX in range(x - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '红' in state_list[toY][toX]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break
                    hits = False
                    for toX in range(x + 1, 9):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '红' in state_list[toY][toX]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break

                    toX = x
                    hits = False
                    for toY in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '红' in state_list[toY][toX]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break
                    hits = False
                    for toY in range(y + 1, 10):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '红' in state_list[toY][toX]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break

                # Legal moves for red cannon
                elif state_list[y][x] == '红炮' and current_player_color == '红':
                    toY = y
                    hits = False
                    for toX in range(x - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '黑' in state_list[toY][toX]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break
                    hits = False
                    for toX in range(x + 1, 9):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '黑' in state_list[toY][toX]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break

                    toX = x
                    hits = False
                    for toY in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '黑' in state_list[toY][toX]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break
                    hits = False
                    for toY in range(y + 1, 10):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '黑' in state_list[toY][toX]:
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
                                break

                # Legal moves for black pawn
                elif state_list[y][x] == '黑兵' and current_player_color == '黑':
                    toY = y - 1
                    toX = x
                    if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='黑'):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    # After crossing the river, pawns can also move horizontally
                    if y < 5:
                        toY = y
                        toX = x + 1
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='黑'):
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        toX = x - 1
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='黑'):
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

                # Legal moves for red pawn
                elif state_list[y][x] == '红兵' and current_player_color == '红':
                    toY = y + 1
                    toX = x
                    if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='红'):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if change_state(state_list, m) != old_state_list:
                            moves.append(m)
                    # After crossing the river, pawns can also move horizontally
                    if y > 4:
                        toY = y
                        toX = x + 1
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='红'):
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        toX = x - 1
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='红'):
                            m = str(y) + str(x) + str(toY) + str(toX)
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)

    # Check if the two generals face each other (no piece in between on the same file)
    if K_x is not None and k_x is not None and K_x == k_x:
        face_to_face = True
        for i in range(K_y + 1, k_y, 1):
            if state_list[i][K_x] != '一一':
                face_to_face = False

    # If face-to-face is possible, add that capturing move
    if face_to_face is True:
        if current_player_color == '黑':
            m = str(k_y) + str(k_x) + str(K_y) + str(K_x)
            if change_state(state_list, m) != old_state_list:
                moves.append(m)
        else:
            m = str(K_y) + str(K_x) + str(k_y) + str(k_x)
            if change_state(state_list, m) != old_state_list:
                moves.append(m)

    moves_id = []
    for move in moves:
        moves_id.append(move_action2move_id[move])
    return moves_id


# Board logic controller
class Board(object):

    def __init__(self):
        self.state_list = copy.deepcopy(state_list_init)
        self.game_start = False
        self.winner = None
        self.state_deque = copy.deepcopy(state_deque_init)

    # Initialize the board
    def init_board(self, start_player=1):   # pass in the starting player's id
        # Build color<->id mappings
        # Red always moves first
        self.start_player = start_player

        if start_player == 1:
            self.id2color = {1: '红', 2: '黑'}
            self.color2id = {'红': 1, '黑': 2}
            self.backhand_player = 2
        elif start_player == 2:
            self.id2color = {2: '红', 1: '黑'}
            self.color2id = {'红': 2, '黑': 1}
            self.backhand_player = 1
        # Current side to move (start with the first player)
        self.current_player_color = self.id2color[start_player]     # '红'
        self.current_player_id = self.color2id['红']
        # Reset board state
        self.state_list = copy.deepcopy(state_list_init)
        self.state_deque = copy.deepcopy(state_deque_init)
        # Reset last move
        self.last_move = -1
        # Count ply since last capture
        self.kill_action = 0
        self.game_start = False
        self.action_count = 0   # action counter
        self.winner = None

    @property
    # All legal moves (IDs) for the current position
    def availables(self):
        return get_legal_moves(self.state_deque, self.current_player_color)

    # Return board state from the current player's perspective; shape: [9, 10, 9] (CHW)
    def current_state(self):
        _current_state = np.zeros([9, 10, 9])
        # Use 9 planes to encode the board:
        # planes 0–6: piece positions; 1 for Red, -1 for Black (latest position in the deque)
        # plane 7: the opponent's last move (from square = -1, to square = +1, others = 0)
        # plane 8: whether current player is the first player (all ones if yes, else zeros)
        _current_state[:7] = state_list2state_array(self.state_deque[-1]).transpose([2, 0, 1])  # [7, 10, 9]

        if self.game_start:
            # Decode self.last_move
            move = move_id2move_action[self.last_move]
            start_position = int(move[0]), int(move[1])
            end_position = int(move[2]), int(move[3])
            _current_state[7][start_position[0]][start_position[1]] = -1
            _current_state[7][end_position[0]][end_position[1]] = 1
        # Indicate which side is to move
        if self.action_count % 2 == 0:
            _current_state[8][:, :] = 1.0

        return _current_state

    # Apply a move to update the board
    def do_move(self, move):
        self.game_start = True  # the game has started
        self.action_count += 1  # increment move counter
        move_action = move_id2move_action[move]
        start_y, start_x = int(move_action[0]), int(move_action[1])
        end_y, end_x = int(move_action[2]), int(move_action[3])
        state_list = copy.deepcopy(self.state_deque[-1])
        # Check capture
        if state_list[end_y][end_x] != '一一':
            # Capturing the opponent's general ends the game immediately
            self.kill_action = 0
            if self.current_player_color == '黑' and state_list[end_y][end_x] == '红帅':
                self.winner = self.color2id['黑']
            elif self.current_player_color == '红' and state_list[end_y][end_x] == '黑帅':
                self.winner = self.color2id['红']
        else:
            self.kill_action += 1
        # Update board state
        state_list[end_y][end_x] = state_list[start_y][start_x]
        state_list[start_y][start_x] = '一一'
        # Switch side to move
        self.current_player_color = '黑' if self.current_player_color == '红' else '红'
        self.current_player_id = 1 if self.current_player_id == 2 else 2
        # Record last move
        self.last_move = move
        self.state_deque.append(state_list)

    # Whether there is a winner
    def has_a_winner(self):
        """Three possible outcomes: Red wins, Black wins, or draw"""
        if self.winner is not None:
            return True, self.winner
        elif self.kill_action >= CONFIG['kill_action']:  # draw → first player loses
            # return False, -1
            return True, self.backhand_player
        return False, -1

    # Check if the game has ended
    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif self.kill_action >= CONFIG['kill_action']:  # draw, no winner
            return True, -1
        return False, -1

    def get_current_player_color(self):
        return self.current_player_color

    def get_current_player_id(self):
        return self.current_player_id


# Game controller wrapping Board; runs a full game loop, collects data, and can render the board
class Game(object):

    def __init__(self, board):
        self.board = board

    # Simple text visualization
    def graphic(self, board, player1_color, player2_color):
        print('player1 side: ', player1_color)
        print('player2 side: ', player2_color)
        print_board(state_list2state_array(board.state_deque[-1]))

    # For human vs AI, human vs human, etc.
    def start_play(self, player1, player2, start_player=1, is_shown=1):
        if start_player not in (1, 2):
            raise Exception('start_player should be either 1 (player1 first) or 2 (player2 first)')
        self.board.init_board(start_player)  # initialize the board
        p1, p2 = 1, 2
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        while True:
            current_player = self.board.get_current_player_id()      # id of the side to move (Red starts)
            player_in_turn = players[current_player]                 # agent for the current side
            move = player_in_turn.get_action(self.board)             # agent returns a move
            self.board.do_move(move)                                 # apply the move
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if winner != -1:
                    print("Game end. Winner is", players[winner])
                else:
                    print("Game end. Draw")
                return winner


    # Use MCTS to start self-play; store (state, MCTS move probabilities, outcome) triplets for NN training
    def start_self_play(self, player, is_shown=False, temp=1e-3):
        self.board.init_board()     # initialize board, start_player=1
        p1, p2 = 1, 2
        states, mcts_probs, current_players = [], [], []
        # Begin self-play
        _count = 0
        while True:
            _count += 1
            if _count % 20 == 0:
                start_time = time.time()
                move, move_probs = player.get_action(self.board,
                                                     temp=temp,
                                                     return_prob=1)
                print('Time for this move: ', time.time() - start_time)
            else:
                move, move_probs = player.get_action(self.board,
                                                     temp=temp,
                                                     return_prob=1)
            # Save self-play data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player_id)
            # Execute one move
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                # For each state, record the outcome from the perspective of the player to move
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0
                # Reset MCTS root
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is:", winner)
                    else:
                        print('Game end. Draw')

                return winner, zip(states, mcts_probs, winner_z)


if __name__ == '__main__':
    # Test array2string
    # _array = np.array([0, 0, 0, 0, 0, 0, 0])
    # print(array2num(_array))

    """# Test change_state
    new_state = change_state(state_list_init, move='0010')
    for row in range(10):
        print(new_state[row])"""

    """# Test print_board
    _state_list = copy.deepcopy(state_list_init)
    print_board(state_list2state_array(_state_list))"""

    """# Test get_legal_moves
    moves = get_legal_moves(state_deque_init, current_player_color='黑')
    move_actions = []
    for item in moves:
        move_actions.append(move_id2move_action[item])
    print(move_actions)"""

    # Test Board.start_play
    # class Human1:
    #     def get_action(self, board):
    #         # print('player1 acting now')
    #         # print(board.current_player_color)
    #         # move = move_action2move_id[input('Please input')]
    #         move = random.choice(board.availables)
    #         return move
    #
    #     def set_player_ind(self, p):
    #         self.player = p
    #
    #
    # class Human2:
    #     def get_action(self, board):
    #         # print('player2 acting now')
    #         # print(board.current_player_color)
    #         # move = move_action2move_id[input('Please input')]
    #         move = random.choice(board.availables)
    #         return move
    #
    #     def set_player_ind(self, p):
    #         self.player = p
    #
    # human1 = Human1()
    # human2 = Human2()
    # game = Game(board=Board())
    # for i in range(20):
    #     game.start_play(human1, human2, start_player=2, is_shown=0)
    board = Board()
    board.init_board()




