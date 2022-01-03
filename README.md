# Deep Reinforcement Learning. Generalization in Video Games.

The goal of the project is to teach the agent how to play multiple video games using PPO algorithm in the Procgen environment. The agent should be able run in many different environments by that simulate human behavior. People can adjust their thinking according to new situations but can machines do that? In this project we explored this problem to see if machines can learn to generalize. We buildt a model with the Proximal Policy Optimization algorithm, and trained it in OpenAI’s procgen environment. 

Procgen Benchmark consists of 16 procedurally-generated environments (games) - bigfish, bossfight, caveflyer, chaser, climber, coinrun, dodgeball, fruitbot, heist, jumper, leaper, maze, miner, ninja, plunder, starpilot. Each observation size is 64x64x3. The agent can choose between 15 possible actions at every step of the game. The Procgen Benchmark was especially designed to explore the problem of generalization in Deep Reinforcement Learning.


