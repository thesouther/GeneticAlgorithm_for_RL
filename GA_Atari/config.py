#-*- coding: utf-8 -*-
import numpy as np 

class Config:
    def __init__(self):
        self.observation_space = 0 # 状态空间，由环境定义时指定
        self.action_space = 0 # 动作空间维度，由环境定义时指定
        self.num_hidden_nodes = 32 # 使用两层网络时，隐层的权重数量 
        
        self.population_size = 100 # 每一代群体中的个体数量，应该>=5
        self.train_generations = 150 # 训练迭代的代数
        
        self.crossover_rate = [0.3,0.6]
        self.crossover_alpha = 0.75

        self.select_rate = 0.6   # 
        self.lucky_select_rate = 0

        self.mutation_rate = 0.2
        self.mutation_alpha = np.random.rand()

        self.elitism = True  # 是否使用精英政策

        self.save_path = './ckpt/individual.pkl'

    def set_params(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.genome_dim = (self.observation_space * self.num_hidden_nodes + 
                          1 * self.num_hidden_nodes +
                          self.num_hidden_nodes * self.action_space +
                          1 * self.action_space)  # 染色体维度
