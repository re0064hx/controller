import gym
env = gym.make('CartPole-v0')

def main():

    observation = env.reset()
    for t in range(1000):
        env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode{} finished after {} timesteps".format(i, t+1))
            break
env.close()


if __name__ == "__main__":
    main()