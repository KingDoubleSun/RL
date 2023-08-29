import numpy as np
import time
import keyboard


class Game:
    def __init__(self, games) -> None:
        self.state = 0
        self.round = 0
        self.games = games
        self.wins = 0
        self.terminated = False
    
    
    def _increase_round(self) -> None:
        if self.round == len(self.games):
            self.terminated = True
            return
        self.state = self.games[self.round]
        self.round += 1
    
    def step(self, action):
        if action == self.state:
            reward = 0
        elif action == 0 and self.state == 1:
            reward = 1
            self.wins += 1
        elif action == 0 and self.state == 2:
            reward = -1
        elif action == 1 and self.state == 0:
            reward = -1
        elif action == 1 and self.state == 2:
            reward = 1
            self.wins += 1
        elif action == 2 and self.state == 0:
            reward = 1
            self.wins += 1
        elif action == 2 and self.state == 1:
            reward = -1
        self._increase_round()
        return self.round - 1, reward, self.terminated

    def init(self):
        self._increase_round()
        return self.round - 1, self.terminated


class Game2:
    def __init__(self, x, y, random_pos_lst, max_steps=20) -> None:
        # random_pos_lst = np.random.choice(x * y, size=2 + hole_num, replace=False)
        self.random_pos_lst = random_pos_lst
        self.x = x
        self.y = y
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_pos = (random_pos_lst[0] // x, random_pos_lst[0] % x)
        self.reward_pos = (random_pos_lst[1] // x, random_pos_lst[1] % x)
        self.hole_pos_list = [(pos // x, pos % x) for pos in random_pos_lst[2:]]
    
    def init(self):
        return self.agent_pos , 0, False

    
    def step(self, action):
        # Implement the game logic based on the action chosen by the agent
        if action == 0:  # Up
            self.agent_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
        elif action == 1:  # Down
            self.agent_pos = (min(self.x - 1, self.agent_pos[0] + 1), self.agent_pos[1])
        elif action == 2:  # Left
            self.agent_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
        elif action == 3:  # Right
            self.agent_pos = (self.agent_pos[0], min(self.y - 1, self.agent_pos[1] + 1))
        
        # Calculate the reward based on the agent's position and the treasure location
        if self.agent_pos == self.reward_pos:
            reward = 10
        elif self.agent_pos in self.hole_pos_list:
            reward = -10
        else:
            reward = 0

        # Update the current step count
        self.current_step += 1

        # Check if the episode is done (either the agent found the treasure or reached the maximum steps)
        done = self.agent_pos == self.reward_pos or self.current_step >= self.max_steps or self.agent_pos in self.hole_pos_list

        return self.agent_pos, reward, done
        
    
    def render(self):
        # clear
        for _ in range(self.x):
            print('\033[1A', end='\x1b[2K')

        board = np.zeros((self.x, self.y), dtype=np.int8)
        board[self.reward_pos] = 1
        board[self.agent_pos] = 9
        for hole_pos in self.hole_pos_list:
            board[hole_pos] = -1
        
        graph = ''
        for row in board:
            graph += '|'
            for cell in row:
                cell_item = ' '
                if cell == 9:
                    cell_item = 'Y'
                elif cell == 1:
                    cell_item = 'O'
                elif cell == -1:
                    cell_item = 'X'
                graph += cell_item
                graph += '|'
            graph += '\n'
        print(graph, end='\r')




if __name__ == '__main__':
    x, y = 10, 10
    game_board = [0, 98, 7, 10, 11, 12, 67, 23, 55, 45]
    max_steps = 100
    actions = 4
    q_table = np.random.rand(x * y, actions)
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.1
    # env = Game(games)
    # round, terminated = env.init()
    # observation = env.state
    # while not terminated:
    #     action = int(input(f'round {round}, state {observation} input: '))
    #     round, reward, terminated = env.step(action)
    #     observation = env.state
    #     print(reward)
    # print(env.wins)
    # print(q_table)

    def obs_trslt(y, pos):
        return pos[0] * y + pos[1]

    for i in range(500):
        env = Game2(x, y, game_board, max_steps)
        observation, reward, terminated = env.init()
        while not terminated:
            pos = obs_trslt(y, observation)
            current_q = np.max(q_table[pos])
            if np.random.rand() > epsilon:
                action = np.argmax(q_table[pos])
            else:
                action = np.random.randint(0, actions - 1 )
            next_observation, reward, terminated = env.step(action)

            new_q = current_q + learning_rate * (reward + discount_factor * np.max(q_table[obs_trslt(y, next_observation)]) - current_q)

            q_table[pos][action] = new_q

            observation = next_observation
    # print(q_table)
    env = Game2(x, y, game_board, max_steps)
    observation, reward, terminated = env.init()
    env.render()
    time.sleep(1)
    while not terminated:
        pos = obs_trslt(y, observation)
        action = np.argmax(q_table[pos])
        next_observation, reward, terminated = env.step(action)
        env.render()
        time.sleep(1)
        observation = next_observation
    
    # print(q_table)
    # game = Game2(4, 4, [0, 11, 7, 10])
    # done = False
    # while not done:
    #     if keyboard.is_pressed('left'):
    #         reward, done = game.step(2)
    #         time.sleep(0.1)
    #     if keyboard.is_pressed('right'):
    #         reward, done = game.step(3)
    #         time.sleep(0.1)
    #     if keyboard.is_pressed('down'):
    #         reward, done = game.step(1)
    #         time.sleep(0.1)
    #     if keyboard.is_pressed('up'):
    #         reward, done = game.step(0)
    #         time.sleep(0.1)