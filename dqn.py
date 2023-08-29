import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from collections import deque
import time
from tqdm import tqdm
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Game:
    def __init__(self, x, y, random_pos_lst, max_steps=20) -> None:
        # random_pos_lst = np.random.choice(x * y, size=2 + hole_num, replace=False)
        self.random_pos_lst = random_pos_lst
        self.x = x
        self.y = y
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_pos = (random_pos_lst[0] // x, random_pos_lst[0] % x)
        self.reward_pos = (random_pos_lst[1] // x, random_pos_lst[1] % x)
        self.hole_pos_list = [(pos // x, pos % x)
                              for pos in random_pos_lst[2:]]

    def init(self):
        return self.get_board(), 0, False

    def step(self, action):
        previous_pos = self.agent_pos
        # Implement the game logic based on the action chosen by the agent
        if action == 0:  # Up
            self.agent_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
        elif action == 1:  # Down
            self.agent_pos = (
                min(self.x - 1, self.agent_pos[0] + 1), self.agent_pos[1])
        elif action == 2:  # Left
            self.agent_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
        elif action == 3:  # Right
            self.agent_pos = (self.agent_pos[0], min(
                self.y - 1, self.agent_pos[1] + 1))

        # Calculate the reward based on the agent's position and the treasure location
        if self.agent_pos == self.reward_pos:
            reward = 10.0
        elif self.agent_pos in self.hole_pos_list:
            reward = -10.0
        elif self.agent_pos == previous_pos:
            reward = -5.0
        else:
            reward = 0.0

        # Update the current step count
        self.current_step += 1

        # Check if the episode is done (either the agent found the treasure or reached the maximum steps)
        done = self.agent_pos == self.reward_pos or self.current_step >= self.max_steps or self.agent_pos in self.hole_pos_list

        return self.get_board(), reward, done

    def render(self):
#         # clear
#         for _ in range(self.x):
#             print('\033[1A', end='\x1b[2K')

        board = np.zeros((self.x, self.y), dtype=np.int8)
        board[self.reward_pos] = 1
        board[self.agent_pos] = 9
        for hole_pos in self.hole_pos_list:
            board[hole_pos] = 5

        graph = ''
        for row in board:
            graph += '|'
            for cell in row:
                cell_item = ' '
                if cell == 9:
                    cell_item = 'Y'
                elif cell == 1:
                    cell_item = 'O'
                elif cell == 5:
                    cell_item = 'X'
                graph += cell_item
                graph += '|'
            graph += '\n'
        print(graph, end='\r')

    def get_board(self):
        board = np.zeros((self.x, self.y), dtype=np.int8)
        board[self.reward_pos] = 1
        board[self.agent_pos] = 9
        for hole_pos in self.hole_pos_list:
            board[hole_pos] = 5
        return board


if __name__ == '__main__':
    x, y = 10, 10
    learning_rate = 0.01

    policy_network = keras.Sequential(
        [
            keras.Input(shape=(x*y)),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(4, activation='linear')
        ]
    )
    target_network = keras.models.clone_model(policy_network)
    target_network.set_weights(policy_network.get_weights())

    policy_network.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.mean_squared_error)

    
    c = 3
    # game_board = [14, 2]
    max_steps = 100
    num_actions = 4
    action_list = np.array(range(num_actions))
    gamma = 0.9
    batch_size = 64
    exp_size = 300
    epsilon = 0.2
    epsilon_up_factor = 1.1
    epsilin_down_factor = 0.8
    consecutive_wins_lmt = 10

    # (state, action, reward, state_bar)
    experience_replay = deque(maxlen=exp_size)


    # game_board = [0, 52, 67, 23, 55, 45]
    game_board = [55, 22]
    consecutive_wins = 0
    win_episodes = []
    for episode in tqdm(range(500)):
        steps = 0
        env = Game(x, y, game_board, max_steps)
        observation, reward, done = env.init() 
        while not done:
            state_input = observation.reshape(-1, x * y) / 9.0 # normalization
            if np.random.rand() > epsilon:
                action = np.argmax(policy_network.predict(state_input))
            else:
                worst_action = np.argmin(policy_network.predict(state_input))
                mask = action_list != worst_action
                better_action_list = action_list[mask]
                action = np.random.choice(better_action_list, 1).item()
            next_observation, reward, done = env.step(action)
            next_state_input = next_observation.reshape(-1, x * y) / 9.0 # normalization

            experience_replay.append((state_input, action, reward, next_state_input, done))
            
            if done and reward > 0:
                epsilon *= epsilin_down_factor
                win_episodes.append(episode)
                consecutive_wins += 1
            elif done and reward <= 0 and epsilon <= 0.4:
                epsilon *= epsilon_up_factor
                consecutive_wins = 0
            elif done and reward <= 0:
                consecutive_wins = 0
            
            if batch_size <= len(experience_replay):
                memories = random.sample(experience_replay, batch_size)

                states = np.squeeze(np.array([memory[0] for memory in memories]))
                actions = np.array([memory[1] for memory in memories])
                rewards = np.array([memory[2] for memory in memories])
                next_states = np.squeeze(np.array([memory[3] for memory in memories]))
                dones = np.array([memory[4] for memory in memories])
                
                q_values = policy_network.predict(states)
                next_q_values = target_network.predict(next_states)
                
                targets = np.copy(q_values)
                for i in range(batch_size):
                    targets[i, int(actions[i])] = rewards[i] + gamma * np.max(next_q_values[i]) * (1 - dones[i])
                
                policy_network.fit(states, targets, batch_size=2, epochs=1)
                steps += 1
                if steps % c == 0:
                    target_network.set_weights(policy_network.get_weights()) 
            observation = next_observation
        if consecutive_wins >= consecutive_wins_lmt:
            break