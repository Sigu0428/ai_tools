# Applying AI methods to the Ludo board game
Two different methods of reinforcement learning are implemented and evaluated on the LUDO board game: Classic tabular Q-learning, and double deep Q-learning with experience replay.

![Screenshot From 2025-06-22 13-42-43](https://github.com/user-attachments/assets/2519cc36-52cc-4a26-bdee-6876ebc4b9fc)

# Tabular Q-learning
I use an epsilon-greedy policy for chosing actions and update the state-action values according to:

![Screenshot From 2025-06-22 13-45-51](https://github.com/user-attachments/assets/00b4e2a5-11ad-4104-b2b0-ed1712f531dd)

# Double deep Q-learning with expericence replay
Tabular reinforcement learning requires a table stored in memory.
The size of this table grows rapidly with the dimensionality of the state-action space.
Alternatively, a neural network can be used to approximate the Q-table. These methods are called 'deep' reinforcement learning.

As described in Melrose Roderick and James MacGlashan and Stefanie Tellex, "Implementing the Deep Q-Network", CoRR.

There are two tricks that are usefull for learning.
The first is called experience replay. The idea is to keep a memory bank of recent state-actions and rewards and then sample a randomized mini-batch from the memory bank to use for training.
This helps to de-correlate experiences in time, such that the learning is not biased by recent turns or recent games.

The second trick is to use two neural networks to make training more stable.
One network is used to choose actions and updated using training data.
The other network is used to generate expected return targets, and periodically has its weights overwritten by the weights from the other network.
The idea is that if the same network is used for both, then its shooting for a moving target, which makes the training unstable.

![Screenshot From 2025-06-22 13-45-17](https://github.com/user-attachments/assets/61cdeefe-4da9-4422-adfa-a799937f1433)

# Reduced state-action space
In LUDO there are 16 pieces that can each be at 57 different positions. The dice can be between 1 and 6, and you can choose between up to 4 pieces to move, depending on how many are on the board. That means the size of the state-action space is around $5 dot 10^29$. This is problematic for the tabular method, which requires storing a value in memory for each state-action. In fact, its impossible. Instead the state will have to be condensed by preprocessing it. The way I'm doing this is to manually program some of the factors a human might consider before choosing a piece to move. The condensed state contains the following information: 
- Which piece will defeat another piece if chosen?
- Which piece will be deployed to the board if chosen?
- Which piece is in danger, and will no longer be if chosen?
- Which piece will hit the goal if chosen?
- Which piece won't be defeated if chosen?

Each of these is encoded as a number between and including 0 and 4. If there are multiple pieces fulfill the condition the lowest piece number is chosen. The number 4 does not correspond to a piece and signifies that no pieces fulfill the condition. The piece not being defeated if chosen is also required for the other conditions, and is meant as a tie breaker if all other parts of the state are 4.
A piece is classified as 'in danger' if there is another piece within 6 fields of it.

# Results
I created a manually programmed agent that follows a simple priority logic. 
This 'priority agent' will be used as a benchmark for evaluating the two AI methods, and it priorities actions in the following hierarchy: deploy new piece > escape danger posed by enemy piece behind it > defeat an enemy piece to send it home > get a piece into the goal > avoid making a move that would send its own piece home.
I can design the reward function to implicitly contain the same hierarchy.

The agents are rewarded for the following state transitions: Defeating another piece, having a piece defeated, getting a piece home and deploying a piece to the board.
The reward function is designed to mimic the hierarchy in the priority agent: deploying a piece is 80, having its piece defeated is -240, defeating another piece is 40 and getting a piece into the goal is 10.
If the AI methods are able to achieve the same level of performance as the priority agent, then i know the implementation is working.

In games against 3 agents taking random actions the priority agent achieved a win-rate of 0.552% in 1000 games.

To asses the performance of the two methods i did 58 runs of tabular Q-learning and deep Q-learning against 3 agents taking random actions, and averaged the result.
To visualize the variation between runs I've plotted the standard deviation as well.

## Tabular Q-learning:
![Screenshot From 2025-06-22 13-52-17](https://github.com/user-attachments/assets/af4d4ab8-c28f-493a-b0d0-1fd3dbce56b3)

## Double deep q-learning with experience replay:
![Screenshot From 2025-06-22 13-51-41](https://github.com/user-attachments/assets/713bbbaa-5c14-4185-8b6e-17ac02186ac8)

### Conclusion
As shown, neither algorithm was able to attain the same performance as the priority agent on average.
I believe this is caused by a learning rate that's too high.
They are not able to do better than the priority agent because its policy is probably close to optimal, given the reduced state-action space.

The methods were not able to learn if they only received a reward when winning, which would rely more on bootstrapping, because rewards were too sparse.
Therefore they need rewards for e.g. taking other pieces and avoiding dangerous positions.
However, any reward that's not from winning will bias the agent into certain behavior, which limits its attainable win rate no matter how long it trains.

An obvious next step would be to try to apply the deep Q-learning method to a larger state-action space, to see if it could achieve even better performance when it has access to more information.
I believe this could also allow for a more sparse reward function, because the relation between state-action and next state is tighter the more information the agent has about the game state.
