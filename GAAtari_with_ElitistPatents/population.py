#-*- coding: utf-8 -*-
import numpy as np
import copy 
import time 
import random

#=====================个体==========================
class Individual:
    '''Individual: 表示个体，这里一个个体模拟一个智能体玩游戏'''
    def __init__(self,conf):
        '''
        ranges：解向量（染色体）数据范围；这里是权重w的范围，(-1,1)
        genome：染色体，这里是权重向量
        fitness： 适应度函数，使用reward，也可以用Q值，后续考虑
        '''
        self.conf = conf
        # 染色体向量
        self.ranges=[-1,1]
        self.genome_dim = self.conf.genome_dim
        dna = np.random.uniform(self.ranges[0],self.ranges[1], [self.genome_dim])
        self.set_genome_weight(dna)

        self.fitness = 0
         
    def set_genome_weight(self, dna):
        self.genome = dna
        len_hidden_w = self.conf.observation_space * self.conf.num_hidden_nodes
        start_output_w = len_hidden_w + 1 * self.conf.num_hidden_nodes
        start_output_b = start_output_w + self.conf.num_hidden_nodes * self.conf.action_space 
        end_output_b = start_output_b + 1 * self.conf.action_space
        self.h_w = np.reshape(self.genome[0:len_hidden_w], [self.conf.observation_space, self.conf.num_hidden_nodes])
        self.h_b = np.reshape(self.genome[len_hidden_w : start_output_w], [1,self.conf.num_hidden_nodes])
        self.o_w = np.reshape(self.genome[start_output_w : start_output_b], [self.conf.num_hidden_nodes, self.conf.action_space])
        self.o_b = np.reshape(self.genome[start_output_b :end_output_b], [1, self.conf.action_space])

    def cal_next_move(self, obsv): # 可以看作evaluation函数，使用两层网络计算，其权重为genome
        x = np.array(obsv).reshape([1,self.conf.observation_space])
        x = self.sigmoid(np.dot(x, self.h_w) + self.h_b)
        output = self.sigmoid(np.dot(x, self.o_w) + self.o_b)
        # print("action",np.argmax(output))
        return np.argmax(output)

    def play_and_evaluation(self, env, num_games=1, visualization=False):
        self.fitness = 0
        for game in range(num_games):
            env.reset()
            pre_observation = []
            while True:
                if visualization:
                    env.render()
                    # time.sleep(0.01)
                if pre_observation == []:
                    step = np.random.randint(0, self.conf.action_space) 
                else:
                    step = self.cal_next_move(pre_observation)
                # print(step)
                observation, reward, done, _ = env.step(step)
                pre_observation = observation
                self.fitness += reward # 用环境回报作为适应度
                if done:
                    pre_observation = []
                    break
        return self.fitness/num_games


    def sigmoid(self, x):
        return 1/1/(1+np.exp(np.negative(np.abs(x))))

    # def cal_weights(self):
    #     self.

#=====================种群==========================
class Population:
    def __init__(self, individual,conf):
        '''
        individual: 个体
        size: 个体数量
        '''
        self.conf = conf
        self.individual = individual
        self.size = self.conf.population_size
        self.individuals = None

    def initialize(self):
        '''初始化下一代'''
        IndvClass = self.individual.__class__
        self.individuals = np.array([IndvClass(self.conf) for i in range(self.size)], dtype=IndvClass)
        # print(self.individuals.shape)

    def fitness(self, env, num_games=1, visualization=True):
        '''
        为每个个体计算适应度，并对适应度进行归一化处理
        为了之后的选择提供方便
        '''
        fitness_list = []
        for I in self.individuals:
            fitness = I.play_and_evaluation(env, num_games=num_games, visualization=visualization)
            fitness_list.append(fitness)
        fitness_list /= np.sum(fitness_list)

        best_pos = np.argmax(fitness_list)
        best_individual = self.individuals[best_pos]
        return fitness_list, best_individual

#=====================选择==========================
class Selection:
    '''选择操作的基类'''
    def select(self, population, select_rate,lucky_select_rate):
        raise NotImplementedError
        
class RouletteWheelSelection(Selection):
    '''
    用轮盘赌选择群体  
    群体中使用适应度函数选择个体 
    '''
    def select(self, population,select_rate,lucky_select_rate):
        sorted_pop = sorted(population.individuals, key=lambda individual: individual.fitness, reverse=True)
        # selected_individuals = np.random.choice(population.individuals, population.size, p=fitness)
        selected_individuals = []
        for i in range(int(select_rate*population.size)):
            selected_individuals.append(sorted_pop[i])
        for i in range(int(lucky_select_rate*population.size)):
            ran = np.random.randint(select_rate*population.size, population.size)
            selected_individuals.append(sorted_pop[ran])
        random.shuffle(selected_individuals)
        population.individuals = np.array([copy.deepcopy(I) for I in selected_individuals])

#=====================交叉==========================
class Crossover:
    def __init__(self, conf):
        '''
        rate: 交叉概率 
        alpha: 交叉时乘的插值
        '''
        self.conf = conf
        self.rate = conf.crossover_rate
        self.alpha = conf.crossover_alpha
        
    @staticmethod
    def cross_individuals(individual_a, individual_b, alpha,conf):
        '''交叉操作
        alpha: 线性插值银因子，当alpha=0.0， 两个基因整体交换，否则交换部分基因
        '''
        pos = np.random.rand(individual_a.genome_dim) <= 0.5

        temp = (individual_b.genome - individual_a.genome)*pos * (1-alpha)
        new_value_a = individual_a.genome + temp
        new_value_b = individual_b.genome - temp

        new_individual_a = Individual(conf)
        new_individual_b = Individual(conf)

        new_individual_a.set_genome_weight(new_value_a)
        new_individual_b.set_genome_weight(new_value_b)

        return new_individual_a, new_individual_b
        
    def cross(self, population):
        adaptive = isinstance(self.rate, list)
        if adaptive:
            fitness = [I.fitness for I in population.individuals]
            fit_max, fit_avg = np.max(fitness), np.mean(fitness)
            
        new_individuals = []
        random_population = np.random.permutation(population.individuals)
        num = int(population.size/2.0)+1

        for individual_a,individual_b in zip(population.individuals[0:num+1], random_population[0:num+1]):
            for _ in range(self.conf.num_children_per_couple):
                if adaptive:
                    fit = max(individual_a.fitness, individual_b.fitness)
                    if fit_max-fit_avg:
                        i_rate = self.rate[1] if fit<fit_avg else self.rate[1]-(self.rate[1]-self.rate[0])*(fit-fit_avg)/(fit_max-fit_avg)
                    else:
                        i_rate = (self.rate[0]+self.rate[1])/2.0
                    if i_rate < 0:
                        i_rate = (self.rate[0]+self.rate[1])/2.0
                else:
                    i_rate = self.rate
                # print("i_rate",i_rate)
                    
                if np.random.rand() <= i_rate:
                    child_individuals = self.cross_individuals(individual_a, individual_b, self.alpha, self.conf)
                    new_individuals.extend(child_individuals)
                else: 
                    new_individuals.append(individual_a)
                    new_individuals.append(individual_b)

        # random.shuffle(new_individuals)
        population.individuals = np.array(new_individuals[0: population.size])
        # print(population.individuals)

#=====================变异==========================
class Mutation:
    def __init__(self, rate):
        self.rate = rate
        
    def mutate_individual(self, individual, positions, alpha):
        '''
        positions: 变异位置， list 
        alpha： 变异量
        '''
        lower_bound = np.ones(individual.genome_dim) * individual.ranges[0]
        upper_bound = np.ones(individual.genome_dim) * individual.ranges[1]
        for pos in positions:
            if np.random.rand() < 0.5:
                individual.genome[pos] -= (individual.genome[pos]-lower_bound[pos])*alpha
            else:
                individual.genome[pos] += (upper_bound[pos]-individual.genome[pos])*alpha

        individual.set_genome_weight(individual.genome)
        individual.fitness = 0
        
    def mutate(self, population, alpha):
        '''alpha： 变异量'''
        for individual in population.individuals:
            if np.random.rand() > self.rate:
                continue
            # print(individual)
            num = np.random.randint(individual.genome_dim)+1
            pos = np.random.choice(individual.genome_dim, num, replace=False)
            self.mutate_individual(individual, pos, alpha)
    
def test():
    import gym
    from config import Config
    conf = Config()

    env = gym.make("Pong-ram-v0").env
    observation_space = env.observation_space.shape[0] #Box(128,)
    action_space = env.action_space.n  # Discrete(6)
    conf.set_params(observation_space=observation_space, action_space=action_space)
    print("conf.observation_space",conf.observation_space)
    print("conf.action_space",conf.action_space)
    print("conf.genome_dim",conf.genome_dim)

    I = Individual(conf)
    # I.cal_next_move()
    # I.play_and_evaluation(env, num_games=1, visualization=True)
    # print("I.h_w",I.h_w.shape) # (128, 20)
    # print("I.h_b",I.h_b.shape) # (1, 20)
    # print("I.o_w",I.o_w.shape) # (20, 6)
    # print("I.o_b",I.o_b.shape) # (1, 6)
    # print("I.fitness",I.fitness)

    P = Population(I, conf)
    P.initialize()
    print("initial P.individuals.shape",P.individuals.shape)
    S = RouletteWheelSelection()
    C = Crossover(conf)

    M = Mutation(conf.mutation_rate)
    # g = GA(P, S, C, M)

    for n in range(1, conf.train_generations+1):    
        print("="*25,n,"="*25)
        for i in P.individuals:
            print("initial i.genome",i.h_b)
        fitness_list, best_individual = P.fitness(env, num_games=1, visualization=True)
        print("fitness P.individuals.shape",P.individuals.shape)
        print('fitness_list, best_individual.fitness ',fitness_list.shape, best_individual.fitness)
        for i in P.individuals:
            print("fitness i.genome",i.h_b)
        S.select(P, conf.select_rate, conf.lucky_select_rate)
        print("selected P.individuals.shape",P.individuals.shape)

        C.cross(P)
        print("cross P.individuals.shape",P.individuals.shape)
        # print("P.individuals.shape",P.individuals.shape)
        for i in P.individuals:
            print("cross i.genome",i.h_b)

        M.mutate(P, conf.mutation_alpha)
        print("mutate P.individuals.shape",P.individuals.shape)
        for i in P.individuals:
            print("mutation i.genome",i.h_b)

        # 精英策略，保留最好的进入下一代
        pos = np.random.randint(P.size)
        P.individuals[pos] = best_individual

    env.close()
    
if __name__ == "__main__":
    test()