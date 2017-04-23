#Observations
import gym
env = gym.make('CartPole-v0')
env2 = gym.make('MountainCar-v0')

for i_episode in range(20):
    observation = env.reset()
    observation2 = env2.reset()

    for t in range(100):
        env.render()
        env2.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        action2 = env2.action_space.sample()
        observation2, reward, done, info = env2.step(action2)

        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

