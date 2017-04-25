import random
 
class Cheese_manu_eqip:
    """チーズ製造機
 
    状態
        s = 0 # 電源 off
        s = 1 # 電源 on
    行動
        a = 0 # 電源ボタンを押す
        a = 1 # 製造ボタンを押す。電源 on の場合のみ製造できる
    報酬
        r = 0 # チーズが出ない
        r = 10 # チーズが出た
    """
 
    def __init__(self):
        self.a_len = 2
        self.s_len = 2
        self.s_init = 0
 
    def action(self, s, a):
        """s で a した時の、次の s と 報酬"""
        if a == 0:
            return 1 - s, 0
        else:
            if s==0:
                return 0, 0
            else:
                return 1, 10
#            return 0, 0 if s == 0 else 10
 



class Agent:
    """エージェント"""
 
    def __init__(self, env, alpha, gamma, epsilon):
        """
 
        env: 環境
        alpha: 学習率
        gamma: 割引率
        epsilon: e-greedy 法のパラメータ
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Qtable = [[0 for i in range(env.a_len)] for j in range(env.s_len)]

    def max_q_action(self, s):
        """s における 最大の Q 値をとる行動"""
        max_q = max(self.Qtable[s])
#        print("max_q:",max_q)
        candidate = [i for i, q in enumerate(self.Qtable[s]) if q == max_q]
#        print("candidate:",candidate)
        return random.choice(candidate)
 
    def eps_greedy(self, s):
        """s において e-greedy 法を用いて次の行動を選択する"""
        if random.random() < self.epsilon:
            return random.randint(0, self.env.a_len - 1)
        else:
            return self.max_q_action(s)
 
    def learn(self):
        """行動して Q 値を更新する"""
        a = self.eps_greedy(self.s)
#        print("a:",a)
        next_s, reward = self.env.action(self.s, a)
#        print("s:",self.s,"next_s:",next_s,"reward:",reward)
    
        self.Qtable[self.s][a] =\
            (1 - self.alpha) * self.Qtable[self.s][a] +\
            self.alpha * (reward + self.gamma * max(self.Qtable[next_s]))
#        print("self.Qtable:",self.Qtable)

        self.s = next_s
 
    def do(self, step):
        """step 回 learn() を行って学習を行う"""
        self.s = self.env.s_init
        for i in range(step):
            self.learn()
 
 
if __name__ == "__main__":

    agent_vm = Agent(Cheese_manu_eqip(), 0.5, 0.9, 0.0001)
    print(agent_vm.Qtable)
    for i in range(50):
        agent_vm.do(3)
    print(agent_vm.Qtable, "\n")

