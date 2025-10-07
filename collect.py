"""Self-play data collection pipeline"""

import random
from collections import deque
import copy
import os
import pickle
import time
from game import Board, Game, move_action2move_id, move_id2move_action, flip_map
from mcts import MCTSPlayer
from config import CONFIG

if CONFIG['use_redis']:
    import my_redis, redis

import zip_array

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('The selected framework is not supported yet.')


class CollectPipeline:
    """Main pipeline for collecting self-play data."""

    def __init__(self, init_model=None):
        # Chessboard logic and controller
        self.board = Board()
        self.game = Game(self.board)
        # Self-play parameters
        self.temp = 1  # temperature
        self.n_playout = CONFIG['play_out']  # number of simulations per move
        self.c_puct = CONFIG['c_puct']  # UCB weight
        self.buffer_size = CONFIG['buffer_size']  # replay buffer size
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        if CONFIG['use_redis']:
            self.redis_cli = my_redis.get_redis_cli()

    def load_model(self):
        """Load the latest or initial model from disk."""
        if CONFIG['use_frame'] == 'paddle':
            model_path = CONFIG['paddle_model_path']
        elif CONFIG['use_frame'] == 'pytorch':
            model_path = CONFIG['pytorch_model_path']
        else:
            print('The selected framework is not supported.')
        try:
            self.policy_value_net = PolicyValueNet(model_file=model_path)
            print('Loaded the latest model successfully.')
        except:
            self.policy_value_net = PolicyValueNet()
            print('Loaded the initial model.')
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1
        )

    def get_equi_data(self, play_data):
        """Perform horizontal flipping to augment data (double dataset size)."""
        extend_data = []
        # Each entry: (state [9,10,9], move probabilities, winner)
        for state, mcts_prob, winner in play_data:
            # Original data
            extend_data.append(zip_array.zip_state_mcts_prob((state, mcts_prob, winner)))
            # Flipped horizontally
            state_flip = state.transpose([1, 2, 0])
            state = state.transpose([1, 2, 0])
            for i in range(10):
                for j in range(9):
                    state_flip[i][j] = state[i][8 - j]
            state_flip = state_flip.transpose([2, 0, 1])
            mcts_prob_flip = copy.deepcopy(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                mcts_prob_flip[i] = mcts_prob[move_action2move_id[flip_map(move_id2move_action[i])]]
            extend_data.append(zip_array.zip_state_mcts_prob((state_flip, mcts_prob_flip, winner)))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """Collect data through self-play."""
        for i in range(n_games):
            self.load_model()  # load the latest model
            winner, play_data = self.game.start_self_play(
                self.mcts_player, temp=self.temp, is_shown=False
            )
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # Data augmentation
            play_data = self.get_equi_data(play_data)
            if CONFIG['use_redis']:
                while True:
                    try:
                        for d in play_data:
                            self.redis_cli.rpush('train_data_buffer', pickle.dumps(d))
                        self.redis_cli.incr('iters')
                        self.iters = self.redis_cli.get('iters')
                        print("Data stored successfully.")
                        break
                    except:
                        print("Data storage failed. Retrying...")
                        time.sleep(1)
            else:
                if os.path.exists(CONFIG['train_data_buffer_path']):
                    while True:
                        try:
                            with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                                data_file = pickle.load(data_dict)
                                self.data_buffer = deque(maxlen=self.buffer_size)
                                self.data_buffer.extend(data_file['data_buffer'])
                                self.iters = data_file['iters']
                                del data_file
                                self.iters += 1
                                self.data_buffer.extend(play_data)
                            print('Successfully loaded existing data.')
                            break
                        except:
                            time.sleep(30)
                else:
                    self.data_buffer.extend(play_data)
                    self.iters += 1
            data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
            with open(CONFIG['train_data_buffer_path'], 'wb') as data_file:
                pickle.dump(data_dict, data_file)
        return self.iters

    def run(self):
        """Start continuous data collection."""
        try:
            while True:
                iters = self.collect_selfplay_data()
                print('batch i: {}, episode_len: {}'.format(iters, self.episode_len))
        except KeyboardInterrupt:
            print('\n\rQuit.')


collecting_pipeline = CollectPipeline(init_model='current_policy.model')
collecting_pipeline.run()

if CONFIG['use_frame'] == 'paddle':
    collecting_pipeline = CollectPipeline(init_model='current_policy.model')
    collecting_pipeline.run()
elif CONFIG['use_frame'] == 'pytorch':
    collecting_pipeline = CollectPipeline(init_model='current_policy.pkl')
    collecting_pipeline.run()
else:
    print('The selected framework is not supported.')
    print('Training finished.')
