#Spaces
#https://github.com/openai/gym/wiki/CartPole-v0

import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print("action_space:", env.action_space)
        print("observation_space:",env.observation_space)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

