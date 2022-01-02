import numpy as np

import maze
import  maze
from RL_brain import DeepQNetwork

rewa=np.zeros(2000)
env = maze.Maze()
def run_maze():
    step = 0
    for episode in range(2000):
        reward_sum = 0
        # initial observation
        observation = env.reset()


        while True:
            # fresh env
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            reward_sum +=reward

            RL.store_transition(observation, action, reward, observation_)

            #if (step > 200) and (step % 5 == 0):
            if RL.memory_counter>RL.memory_size:
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

        print("episode",episode,", reward: ",reward_sum)
        rewa[episode]=reward_sum

    # end of game
    print('game over')






import matplotlib.pyplot as plt

if __name__ == "__main__":
    # maze game
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      e_greedy_increment=0.05
                      )
    run_maze()
    RL.plot_cost()

    xlable = np.arange (1, 51)
    ave = np.zeros (50)
    for i in range (50):
        sum = 0
        for j in range (40 * i, 40 * (i + 1)):
            sum += rewa[j]
        ave[i] = sum / 40.0
    plt.xlabel ("episode")
    plt.ylabel ("reward")
    plt.plot (xlable, ave)
    plt.show ()
