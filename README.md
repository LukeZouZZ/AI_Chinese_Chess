# Build Your Own Xiangqi (Chinese Chess) AI with AlphaZero

## 1) What each file does
- collect.py — Self-play for data collection  
- train.py — Model training  
- game.py — Xiangqi game logic  
- mcts.py — Monte Carlo Tree Search  
- paddle_net.py, pytorch_net.py — Neural networks to evaluate moves  
- play_with_ai.py — Human vs. AI (print/CLI version)  
- UIplay.py — GUI for human vs. AI  

## 2) Two training backends
- Use the PyTorch version: set CONFIG['use_frame'] = 'pytorch' in config.py  
- Use the Paddle version: set CONFIG['use_frame'] = 'paddle' in config.py  

Regardless of the framework, install the GPU version and use an NVIDIA GPU.  
Each MCTS move triggers thousands of NN inferences, so performance is crucial.

## 3) Multi-process synchronous training
- Start self-play (you can run multiple processes):
  python collect.py  
- Start training (run only one process):
  python train.py
