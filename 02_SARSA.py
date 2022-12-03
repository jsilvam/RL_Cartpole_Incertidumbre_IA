import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import trange


#problem = "CartPole-v0"
problem = "CartPole-v1"
rewards = []

class CartPoleAgent():
    def __init__(self, buckets=(1, 1, 6, 12), num_episodes=500, min_lr=0.1, min_epsilon=0.1, discount=0.98, decay=25):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        self.env = gym.make(problem)

        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2],
                             math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2],
                             -math.radians(50) / 1.]

        self.sarsa_table = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize_state(self, obs):
        discretized = list()
        for i in range(len(obs)):
            scaling = (obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def choose_action(self, state):
        #print(np.random.random(), self.epsilon)
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.sarsa_table[state])

    def choose_action_run(self, state, sample=False):
        return np.argmax(self.sarsa_table[state])

    def update_sarsa(self, state, action, reward, new_state, new_action):
        self.sarsa_table[state][action] += self.learning_rate * (
                    reward + self.discount * (self.sarsa_table[new_state][new_action]) - self.sarsa_table[state][
                action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self):
        print("------Training final states:------")
        i=1
        reward_arr = []
        for e in trange(self.num_episodes):
            reset = self.env.reset()
            current_state = self.discretize_state(reset[0])

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False
            truncated = False
            j=0
            rew = 0
            while not (done or truncated):
                action = self.choose_action(current_state)
                obs, reward, done, truncated, info = self.env.step(action)
                new_state = self.discretize_state(obs)
                new_action = self.choose_action_run(new_state)
                self.update_sarsa(current_state, action, reward, new_state, new_action)
                current_state = new_state
                j+=1
                rew += reward
            i+=1
            rewards.append(j)
            reward_arr.append(rew)
        print('Finished training!')
        print("average reward per episode :", sum(reward_arr) / len(reward_arr))

    def run(self, executions=100):
        reward_arr = []
        for i in range(executions):
            self.env = gym.make(problem)#, render_mode="human")
            t = 0
            done = False
            truncated = False
            reset = self.env.reset()
            current_state = self.discretize_state(reset[0])
            rew = 0
            print("------Testing - Execution",i,"all states:------")
            while not (done or truncated):
                self.env.render()
                t = t + 1
                action = self.choose_action_run(current_state)  # Use value algorithm
                obs, reward, done, truncated, info = self.env.step(action)
                #      "Observation: " + str(obs) +
                #      ", Reward: " + str(reward) +
                #      ", Done: " + str(done))
                new_state = self.discretize_state(obs)
                current_state = new_state
                rew += reward
            print("{:4}".format(str(i+1) + ")"), "Observation: " + str(obs) +
                  ", Reward: " + str(reward) +
                  ", Done: " + str(done) +
                  ", Truncated: " + str(truncated) +
                  ", Iterations: " + str(t))
            rewards.append(t)
            reward_arr.append(rew)
        self.env.close()
        print("average reward per episode :", sum(reward_arr) / len(reward_arr))

agent = CartPoleAgent()

rewards=[]
agent.train()
plt.plot(rewards)
plt.title("Train iterations")
plt.show()

rewards = []
agent.run()
plt.plot(rewards)
plt.title("Test iterations")
plt.show()