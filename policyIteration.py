import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np

class PolicyIterationAgent(object):
    def __init__(self, env, theta=1e-4, gamma=0.5):
        self.theta = theta
        self.gamma = gamma
        self.state2dic, self.mdp = env.getMDP()
        self.stateNotDone = self.mdp.keys()
        self.dic2state = {v: k for k, v in self.state2dic.items()}
        self.num_action = 4
        self.num_state = len(self.state2dic)

        # For policy taking the action
        self.policy = {self.state2dic[k]: np.random.randint(self.num_action) for k in self.stateNotDone }
        self.value_function = {self.state2dic[k]: np.random.rand() for k in self.stateNotDone }

        # Getting the optimal Policy
        self.__policyOptimization()
        print(self.policy)


    def __policyOptimization(self):
        old_policy = {k: -1 for k in self.policy.keys()}
        
        while self.policy != old_policy:
            # Policy Evaluation
            self.value_function = self.__policyEvaluation()

            # Policy Improvement
            old_policy = self.policy
            self.policy = self.__policyImprovement()


    def __policyEvaluation(self):
        delta = 2 * self.theta
        while delta > self.theta:
            delta = 0
            for s in self.value_function.keys():
                old_value = self.value_function[s]
                aux = 0
                for p in self.mdp[self.dic2state[s]][self.policy[s]]:
                    if p[3] == True:
                        aux += p[0] * p[2]
                    else:
                        aux += p[0] * ( p[2] + self.gamma * self.value_function[self.state2dic[p[1]]] )

                self.value_function[s] = aux
                delta = max(delta, abs(old_value - self.value_function[s]))
        return self.value_function


    def __policyImprovement(self):
        for s in self.policy.keys():
            aux = np.zeros(self.num_action)
            for action in range(self.num_action):
                for p in self.mdp[self.dic2state[s]][action]:
                    if p[3] == True:
                        aux[action] += p[0] * p[2]    
                    else:
                        aux[action] += p[0] * ( p[2] + self.gamma * self.value_function[self.state2dic[str(p[1])]] )

            self.policy[s] = np.argmax(aux)
        return self.policy


    def act(self, observation):
        return self.policy[self.state2dic[gridworld.GridworldEnv.state2str(observation)]]



if __name__ == "__main__":
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    agent = PolicyIterationAgent(env)
    #teste