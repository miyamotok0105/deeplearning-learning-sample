import gym
#Running an environment

#元のドキュメント
#https://gym.openai.com/docs
#http://qiita.com/masataka46/items/cc37d36137a4a162c04a

env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

