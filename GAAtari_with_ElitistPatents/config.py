#-*- coding: utf-8 -*-
import numpy as np 

class Config:
    def __init__(self):
        self.observation_space = 0 # 状态空间，由环境定义时指定
        self.action_space = 0 # 动作空间维度，由环境定义时指定
        self.num_hidden_nodes = 32 # 使用两层网络时，隐层的权重数量 

        self.num_games_per_individual = 3 # 每个个体在计算fitness时，运行几次游戏
        self.max_fitness = self.num_games_per_individual*21 # fitness的最大值
        
        self.population_size = 100 # 每一代群体中的个体数量，应该>=10
        self.train_generations = 150 # 训练迭代的代数

        self.select_rate = 0.1   # 选择最好的10%的个体繁殖下一代，所以这里要考虑好种群数量应该在100左右
        self.lucky_select_rate = 0.1 # 有一些幸运的个体，被随机选择进行繁殖

        self.crossover_rate = [0.7,0.95]  # 进行交叉繁殖的概率
        self.crossover_alpha = 0.05 # 繁殖时，基因改变量
        self.num_children_per_couple = int((1 / (self.select_rate+self.lucky_select_rate))*2) # 每对夫妻产生的孩子的数量
        # print(self.num_children_per_couple)

        self.mutation_rate = 0.2 # 变异率
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

def test():
    conf = Config()

if __name__ == "__main__":
    test()