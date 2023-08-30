import numpy as np
from collections import deque

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
        return self.get_board() / 9.0, 0, False
    
    def reset(self):
        self.current_step = 0
        self.replay = deque(maxlen=replay_size)
        self.agent_pos = (self.random_pos_lst[0] // self.x, self.random_pos_lst[0] % self.x)
        self.reward_pos = (self.random_pos_lst[1] // x, self.random_pos_lst[1] % x)
        self.hole_pos_list = [(pos // self.x, pos % self.x)
                              for pos in self.random_pos_lst[2:]]
    
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
            reward = 1.0 * (self.max_steps - self.current_step + 1)
        elif self.agent_pos in self.hole_pos_list:
            reward = -100.0
        elif self.agent_pos == previous_pos:
            reward = -50.0
        else:
            reward = 0.0

        # Update the current step count
        self.current_step += 1

        # Check if the episode is done (either the agent found the treasure or reached the maximum steps)
        done = self.agent_pos == self.reward_pos or self.current_step >= self.max_steps or self.agent_pos in self.hole_pos_list

        return self.get_board() / 9.0, reward, done

    def render(self):
#         # clear
#         for _ in range(self.x):
#             print('\033[1A', end='\x1b[2K')
        board = self.get_board()
        graph = ''
        for row in board:
            graph += '|'
            for cell in row:
                cell_item = ' '
                if cell == 1:
                    cell_item = 'Y'
                elif cell == 9:
                    cell_item = 'O'
                elif cell == 5:
                    cell_item = 'X'
                graph += cell_item
                graph += '|'
            graph += '\n'
        print(graph, end='\r')

    def get_board(self):
        board = np.zeros((self.x, self.y), dtype=np.int8)
        board[self.reward_pos] = 9
        board[self.agent_pos] = 1
        for hole_pos in self.hole_pos_list:
            board[hole_pos] = 5
        return board


class Game2:
    def __init__(self, x, y, agent_pos, rewards_lst, holes_lst, max_steps=20, replay_size=3) -> None:
        # random_pos_lst = np.random.choice(x * y, size=2 + hole_num, replace=False)
        self.init_agent_pos = agent_pos
        self.init_rewards_lst = rewards_lst
        self.init_holes_lst = holes_lst
        self.replay_size = replay_size
        self.replay = deque(maxlen=replay_size)
        self.remaining_rewards = len(rewards_lst)
        self.x = x
        self.y = y
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_pos = (agent_pos // x, agent_pos % x)
        self.reward_pos_list = [(pos // x, pos % x)
                           for pos in rewards_lst]
        self.hole_pos_list = [(pos // x, pos % x)
                              for pos in holes_lst]
        
    def init(self):
        result = self.get_board() / 9.0
        for _ in range(self.replay_size):
            self.replay.append(result)
        return self.get_input(), 0, False

    def reset(self):
        self.replay = deque(maxlen=self.replay_size)
        self.remaining_rewards = len(self.init_rewards_lst)
        self.current_step = 0
        self.agent_pos = (self.init_agent_pos // self.x, self.init_agent_pos % self.x)
        self.reward_pos_list = [(pos // self.x, pos % self.x)
                           for pos in self.init_rewards_lst]
        self.hole_pos_list = [(pos // self.x, pos % self.x)
                              for pos in self.init_holes_lst]
        return self.init()
    
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
        if self.agent_pos in self.reward_pos_list:
            self.reward_pos_list.remove(self.agent_pos)
            reward = 20.0 * (len(self.init_rewards_lst) - self.remaining_rewards + 1)
            self.remaining_rewards -= 1
        elif self.agent_pos in self.hole_pos_list:
            reward = -100.0
        elif self.agent_pos == previous_pos:
            reward = -100.0
        else:
            reward = 0.0

        # Update the current step count
        self.current_step += 1

        # Check if the episode is done (either the agent found the treasure or reached the maximum steps)
        done = self.remaining_rewards == 0 or self.current_step >= self.max_steps or self.agent_pos in self.hole_pos_list or self.agent_pos == previous_pos

        if self.agent_pos == previous_pos:
            self.agent_pos = (-1, -1)
        result = self.get_board() / 9.0
        self.replay.append(result)

        return self.get_input(), reward, done

    def render(self):
#         # clear
#         for _ in range(self.x):
#             print('\033[1A', end='\x1b[2K')
        board = self.get_board()
        graph = ''
        for row in board:
            graph += '|'
            for cell in row:
                cell_item = ' '
                if cell == 1:
                    cell_item = 'Y'
                elif cell == 9:
                    cell_item = 'O'
                elif cell == 5:
                    cell_item = 'X'
                graph += cell_item
                graph += '|'
            graph += '\n'
        print(graph, end='\r')

    def get_board(self):
        board = np.zeros((self.x, self.y), dtype=np.int8)
        if self.agent_pos != (-1, -1):
            board[self.agent_pos] = 1 
        for reward_pos in self.reward_pos_list:
            board[reward_pos] = 9
        for hole_pos in self.hole_pos_list:
            board[hole_pos] = 5
        return board
    
    def get_input(self):
        return np.array(self.replay).reshape(-1, self.x * self.y * self.replay_size)
    
