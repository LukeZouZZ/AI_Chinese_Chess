# \# Xiangqi (Chinese Chess) AI with AlphaZero

# 

# \## 1) What each file does

# \- collect.py — Self-play for data collection  

# \- train.py — Model training  

# \- game.py — Xiangqi game logic  

# \- mcts.py — Monte Carlo Tree Search  

# \- paddle\_net.py, pytorch\_net.py — Neural networks to evaluate moves  

# \- play\_with\_ai.py — Human vs. AI (print/CLI version)  

# \- UIplay.py — GUI for human vs. AI  

# 

# \## 2) Two training backends

# \- Use the PyTorch version: set CONFIG\['use\_frame'] = 'pytorch' in config.py  

# \- Use the Paddle version: set CONFIG\['use\_frame'] = 'paddle' in config.py  

# 

# Regardless of the framework, install the GPU version and use an NVIDIA GPU.  

# Each MCTS move triggers thousands of NN inferences, so performance is crucial.

# 

# \## 3) Multi-process synchronous training

# \- Start self-play (you can run multiple processes):

# &nbsp; python collect.py  

# \- Start training (run only one process):

# &nbsp; python train.py

# 

