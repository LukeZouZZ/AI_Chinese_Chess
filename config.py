# Global configuration file

CONFIG = {
    # Framework choice: "pytorch" or "paddle"
    "use_frame": "pytorch",
    # Use Redis distributed mode
    "use_redis": False,
    "redis_host": "127.0.0.1",
    "redis_port": 6379,
    "redis_db": 0,

    # Game & training parameters
    "play_out": 400,           # number of simulations per move
    "c_puct": 5,               # exploration constant
    "batch_size": 512,
    "epochs": 5,
    "kl_targ": 0.02,
    "game_batch_num": 1500,
    "buffer_size": 30000,
    "kill_action": 120,        # draw if no capture in 120 moves
    "dirichlet": 0.3,

    # Paths
    "train_data_buffer_path": "train_data.pkl",
    "paddle_model_path": "current_policy.model",
    "pytorch_model_path": "current_policy.pkl",

    # Training interval in seconds (e.g. 600 = 10 minutes)
    "train_update_interval": 600,
}
