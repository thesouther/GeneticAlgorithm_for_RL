#-*- coding: utf-8 -*-
import numpy as np
import copy 
import gym
from population import Individual,Population,RouletteWheelSelection,Crossover,Mutation

from util import dump_model,load_model

from config import Config
conf = Config()

class GA:
    def __init__(self, population, selection, crossover, mutation):
        self.population = population
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        
    def run(self, env,num_games=1, visualization=True):
        self.population.initialize()
        gen = conf.train_generations
        for n in range(1, gen+1):
            print("="*20,"generations: ",n,"="*20)
            fitness_list, best_individual = self.population.fitness(env, num_games, visualization)
            print("initial P.individuals.shape, best_individual.fitness",self.population.individuals.shape, best_individual.fitness)

            self.selection.select(self.population, conf.select_rate, conf.lucky_select_rate)
            print("selected P.individuals.shape",self.population.individuals.shape)

            self.crossover.cross(self.population)
            print("cross P.individuals.shape",self.population.individuals.shape)

            self.mutation.mutate(self.population, conf.mutation_alpha)
            print("mutate P.individuals.shape",self.population.individuals.shape)

            # 精英策略，保留最好的进入下一代
            if conf.elitism:
                pos = np.random.randint(self.population.size)
                self.population.individuals[pos] = best_individual
            
        return best_individual

def run_train():
    env = gym.make("Pong-ram-v0").env
    observation_space = env.observation_space.shape[0] #Box(128,)
    action_space = env.action_space.n  # Discrete(6)
    conf.set_params(observation_space=observation_space, action_space=action_space)

    I = Individual(conf)
    P = Population(I,conf)
    S = RouletteWheelSelection()
    C = Crossover(conf)
    M = Mutation(conf.mutation_rate)
    g = GA(P, S, C, M)

    best_individual = g.run(env,num_games=1, visualization=True)
    dump_model(best_individual, conf.save_path)
    env.close()

def test_best_ind():
    env = gym.make("Pong-ram-v0").env
    best_individual = load_model(conf.save_path)
    fitness = best_individual.play_and_evaluation(env, num_games=3, visualization=False)
    print("fitness", fitness)
    env.close()

if __name__ == "__main__":
    run_train()
    # test_best_ind()