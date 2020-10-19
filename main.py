import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from policyIteration import PolicyIterationAgent
from randomAgent import RandomAgent
from valueIteration import ValueIterationAgent

if __name__ == "__main__":
    # Environment
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan10.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    # Agent
    # agent = RandomAgent(env)
    # agent = PolicyIterationAgent(env, gamma=0.99)
    agent = ValueIterationAgent(env)


    # Loop
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 200 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()