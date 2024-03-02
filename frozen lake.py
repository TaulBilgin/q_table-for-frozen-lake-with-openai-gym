import numpy as np
import gym
import random

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False)

# Hyperparameters
total_episodes = 10000
learning_rate = 0.1
max_steps = 100
gamma = 0.99

epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.001

# Get action and state space sizes
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

# Initialize Q-table
q_table = np.zeros((state_space_size, action_space_size))
total_reward = 0
episode_count = 0

# Training loop
for episode in range(total_episodes):
    episode_count += 1
    state = env.reset()[0]
    done = False

    for step in range(max_steps):
        # Exploration vs. exploitation
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()

        # Take action and observe new state and reward
        new = env.step(action)
        new_state = new[0]
        reward = new[1]
        done = new[2]

        # Update Q-value using Bellman equation
        q_table[state][action] = (1 - learning_rate) * q_table[state][action] + learning_rate * (reward + gamma * np.max(q_table[new_state]))

        state = new_state
        total_reward += reward

        if done:
            break

    # Decay epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    # Print progress
    print(f"Episode: {episode_count}, Epsilon: {epsilon:.4f}, Total Reward: {total_reward}")

print("\nFinal Q-table:")
print(q_table)