import os
import sys
import argparse
import gym
from agents import RandomAgent, FunFunAgent, TrackAgent
#まだ動いてない

# def main(episode_count):
#     env = gym.make("Pong-v0")
#     action_number = env.action_space.n
#     action_up = 2
#     action_down = 3
#     action_stop = 0

#     # your code here

#     for i in range(episode_count):
#         observation = env.reset()
#         done = False
#         agent = agents()
#         # agent.init()
#         name = "agent"

#         while not done:
#             env.render()
#             action = agent.act(observation)  # agent takes observation, and decide action
#             next_observation, reward, done, info = env.step(action)
#             print("{} takes action={}, and get reward={}.".format(name, action, reward))

#             observation = next_observation
#             agent.reward += reward

#             if done:
#                 print("Episode {}: {} gets total reward={}.".format(i, name, agent.reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's Start OpenAI Gym")
    parser.add_argument("--episode", type=int, help="Episode Count to work", default=2)

    args = parser.parse_args()
    # main(args.episode)

    RandomAgent()

   

