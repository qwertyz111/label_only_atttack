import torch
import numpy as np
from utils_rb import *
from HSJA_rb import HSJA
from SignOPT_rb import OPT_attack_sign_SGD

# main attack
action = -1

import torch.nn as nn

# 定义Net类 (定义网络)

N_ACTIONS = 2               
N_STATES = 20

BATCH_SIZE = 32                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.99                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 2000                          # 记忆库容量

class Net(nn.Module):
    def __init__(self):                                                         # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear(N_STATES, 50)                                      # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.out = nn.Linear(50, N_ACTIONS)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        x = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value 
    
class DQN(object):
    def __init__(self):                                                         # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络:评估网络和目标网络
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 +2 ))             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器(输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)

    def choose_action(self, x):                                                 # 定义动作选择函数 (x为状态)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() < EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)                            # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]                                                  # 输出action的第一个数
        else:                                                                   # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)                            # 这里action随机等于0或1 (N_ACTIONS = 2)
        return action                                                           # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))                                 # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
        self.memory[index, :] = transition                                      # 置入transition
        self.memory_counter += 1                                                # memory_counter自加1

    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1                                            # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]                                 # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()
        
class RamBoAtt():
    def __init__(self,
                model,
                model_ex,
                testset,
                #m_block = 16,  # 1 for CIFAR-10
                #lamda=2,       # 1.2 for CIFAR-10
                #w=1,
                #len_T =1000,   # 500 for CIFAR-10
                #delta = 1e-1,  # 1e-2 for CIFAR-10
                seed = None,
                targeted=True,
                dataset='cifar10'):

        self.model = model
        self.model_ex = model_ex
        self.testset = testset
        #self.m_block = m_block # number of block
        #self.lamda = lamda 
        #self.w = w
        #self.len_T = len_T
        #self.delta = delta
        self.seed = seed
        self.targeted = targeted
        self.dataset=dataset

    def convert_idx_ImgN(self,n, x):
        # c1 = Channel = C
        # c2 = Width = W
        # c3 = Height = H
        # c1 x c2 x c3 = C x W x H
        c1 = n // (x.shape[2] * x.shape[3])
        c2 = (n - c1 * x.shape[2] * x.shape[3])// x.shape[3]
        c3 = n - c1 * x.shape[2] * x.shape[3] - c2 * x.shape[3]
        return c1,c2,c3

    def BlockDescent(self,ori_img, ori_label, target_img, target_label,eps,max_qry,
                     delta,m_block,lamda,w,len_T):
        # With view
        best_adv = target_img.clone()
        D = np.zeros(max_qry+1000)
        wi = ori_img.shape[2]
        he = ori_img.shape[3]
        n_dims = ori_img.view(1, -1).size(1) # ori_img.nelement() = C x W x H  = ori_img.nelement()
        DL = np.inf
        DR = 0
        cnt = 0
        prev_qry = 0
        nqry = 0
        if self.seed != None:
            torch.manual_seed(self.seed)
            
        terminate = False
#         dqn=DQN()
#         state_list=[]
#         action_list=[]
#         reward_list=[]
#         for i in range(20):
#             state_list.append(1)
            
        while not(terminate):
            idx = torch.randperm(n_dims)
            i = 0
            j = 0
            while i < (n_dims-m_block):
                best_adv_temp = best_adv.clone()
                for k in range(m_block):
                    c1,c2,c3 = self.convert_idx_ImgN(idx[i+k], ori_img)
                    
                    '''
                    if (self.w<=c2) & (c2<=wi-self.w) & (self.w<=c3) & (c3<=he-self.w):
                        mask_sign = torch.sign(ori_img[0,c1, c2-self.w:c2+w+1, c3-self.w:c3+self.w+1]-best_adv[0,c1,c2-self.w:c2+self.w+1,c3-w:c3+w+1])
                        best_adv_temp[0,c1,c2-w:c2+w+1,c3-w:c3+w+1] = (best_adv[0,c1,c2-w:c2+w+1,c3-w:c3+w+1] + eps * mask_sign).clamp(0,1)
                        
                    else:
                        w1 = torch.min(torch.tensor([self.w,c2]))
                        w2 = torch.min(torch.tensor([self.w,wi-c2])) + 1
                        h1 = torch.min(torch.tensor([self.w,c3]))
                        h2 = torch.min(torch.tensor([self.w,he-c3])) + 1
                        mask_sign =  torch.sign(ori_img[0, c1, c2-w1:c2+w2, c3-h1:c3+h2]-best_adv[0, c1, c2-w1:c2+w2, c3-h1:c3+h2])
                        best_adv_temp[0, c1, c2-w1:c2+w2, c3-h1:c3+h2] = (best_adv[0, c1, c2-w1:c2+w2, c3-h1:c3+h2] + eps * mask_sign).clamp(0,1)
                    '''
                    
                    w1 = torch.min(torch.tensor([w,c2]))
                    w2 = torch.min(torch.tensor([w,wi-c2])) + 1
                    h1 = torch.min(torch.tensor([w,c3]))
                    h2 = torch.min(torch.tensor([w,he-c3])) + 1
                    mask_sign =  torch.sign(ori_img[0, c1, c2-w1:c2+w2, c3-h1:c3+h2]-best_adv[0, c1, c2-w1:c2+w2, c3-h1:c3+h2])
                    best_adv_temp[0, c1, c2-w1:c2+w2, c3-h1:c3+h2] = (best_adv[0, c1, c2-w1:c2+w2, c3-h1:c3+h2] + eps * mask_sign).clamp(0,1)
                
                if torch.norm(best_adv_temp - ori_img)< torch.norm(best_adv - ori_img):
                    next_pert_lbl = self.model.predict_label(best_adv_temp)
                    #next_pert_lbl = 5
                    #next_pert_lbl = torch.tensor(next_pert_lbl).cuda()
                    next_pert_lbl = random.choice([x for x in range(10) if x!= int(next_pert_lbl)])
            
                    next_pert_lbl = torch.tensor(next_pert_lbl).cuda()
#                     if j<=20:
#                         action_list.append(1)
#                     else:
#                         action_list.append(dqn.choose_action(state_list[j]))
                    

#                     if j<=20:
#                         action_list.append(0)
#                     else:
#                         if dqn.choose_action(np.array(state_list))==1:
#                             random_int = random.choice([x for x in range(10) if x!= int( next_pert_lbl)])
#                             tensor_int = torch.tensor(random_int)
#                             Tlabel = tensor_int.cuda()
#                             action_list.append(1)   
#                         else:
#                             action_list.append(0) 
                    
                    if self.targeted == True:
                        if (next_pert_lbl==target_label):
                            best_adv = best_adv_temp.clone()
                    else:
                        if (next_pert_lbl!=ori_label):
                            best_adv = best_adv_temp.clone()
                    if (nqry%1000)==0:
                        print('Qry#',nqry,'; l2 distance =', torch.norm(best_adv - ori_img).item(),'; adv label:',
                              next_pert_lbl)
                    #self.model.predict_label(best_adv).item()       
                    D[nqry] = torch.norm(best_adv - ori_img)
                    nqry += 1 
                    
                    # control auto terminate
                    if nqry % len_T == 0:
                        DR = np.mean(D[nqry - len_T:nqry])
                        if ((DL-DR) < delta): #
                            terminate = True
                            print('\nBreak due to slow convergence!\n')
                            break
                        else:
                            DL = DR
                if nqry<max_qry:
                    i += m_block
                else:
                    terminate = True
                    print('\nBreak due to exceeding query limit!\n')
                    break

            eps /= lamda
            
#             if action_list[j]==1:
#                 state_list.append(1)
#                 if j >= (len(data_loader)-num_mem):
#                     reward_list.append(5)
#                 else:
#                     reward_list.append(-5)
                
#             else:
#                 state_list.append(0)
#                 if j >= (len(data_loader)-num_mem):
#                     reward_list.append(-5)
#                 else:
#                     reward_list.append(5)
 
#             dqn.store_transition(state_list[0:20],action_list[j],reward_list[j],state_list[1:21])
            
#             state_list.pop(0)
            
#             dqn.learn()
            # engineering to terminate if looping infinitely without any improvement!
            if prev_qry == nqry:
                if cnt ==2:
                    print('Break due to loop infinitely!')
                    break
                else:
                    cnt += 1
            else: 
                cnt = 0
                prev_qry = nqry    
            j+=1
        return best_adv, nqry, D[:nqry]

    def hybrid_attack(self,oimg,olabel,timg,tlabel,query_limit=50000,attack_mode="RBH"):

        # =========== module 1 ===========
        D = np.zeros(query_limit+2000)
        if attack_mode=="RBH":
            if self.targeted:
                y_targ = np.array([tlabel])
            else:
                y_targ = np.array([olabel])

            if self.dataset == 'cifar10':
                delta = 1e-2
                len_T = 500
            elif self.dataset == 'imagenet':
                delta = 1
                len_T = 2000  
            # ========================
            constraint='l2'
            num_iterations=150
            gamma=1.0
            stepsize_search='geometric_progression'
            max_num_evals = 1e4
            init_num_evals=100
            verbose=True
            auto_terminate=True

            module_1 = HSJA(self.model_ex,constraint,num_iterations,gamma,stepsize_search,max_num_evals,init_num_evals, verbose,delta,len_T)
            
            if self.targeted:
                adv, nqry, Dt = module_1.hsja(oimg.cpu().numpy(), y_targ, timg.cpu().numpy(),self.targeted,query_limit,auto_terminate)
            else:
                timg = None
                adv, nqry, Dt = module_1.hsja(oimg.cpu().numpy(), y_targ, timg,self.targeted)
            timg = torch.unsqueeze(torch.from_numpy(adv).float(), 0).cuda()
                    
            print('Module 1: Finished HSJA\n')

        elif attack_mode=='RBS':
            k=200
            if self.dataset == 'cifar10':
                delta = 1e-2
                len_T = 2
            elif self.dataset == 'imagenet':
                delta = 1
                len_T = 4
            auto_terminate=True
            module_1 = OPT_attack_sign_SGD(self.model,k,delta,len_T,self.testset)

            alpha = 0.2
            beta = 0.001
            iterations = 5000
            distortion = None
            stopping = 0.0001
            # ========================

            if self.targeted:
                adv, nqry, Dt = module_1.attack_targeted(oimg, olabel, timg, tlabel, alpha, beta, iterations, query_limit, distortion, self.seed, stopping, auto_terminate)
            else:
                timg = None
                adv, nqry, Dt = module_1.attack_untargeted(oimg, olabel, timg ,alpha, beta, iterations, query_limit, distortion,self.seed, stopping, auto_terminate)

            timg = adv.cuda()
            print('Module 1: Finished SignOPT\n')

        D[:nqry] = Dt
        nquery = nqry

        # =========== module 2 ===========
        if nquery<query_limit:
            max_query = query_limit - nquery
            if self.dataset == 'cifar10':
                pi = 100
                delta = 1e-2
                m_block = 1   
                lamda = 1.2     
                w=1
                len_T = 500 
            elif self.dataset == 'imagenet':
                pi = 50
                delta = 1e-1
                m_block = 16
                lamda = 2     
                w=1
                len_T = 1000 

            eps = np.percentile(torch.abs(timg - oimg).cpu().numpy(), pi)
            adv, nqry, Dt = self.BlockDescent(oimg,olabel,timg,tlabel,eps,max_query,
                                              delta,m_block,lamda,w,len_T)

            print('Module 2: Finished BlockDescent')

            D[nquery:nquery + nqry] = Dt
            nquery += nqry

        # =========== module 3 ===========
        if nquery<query_limit:
            alpha = 0.2
            beta = 0.001
            iterations = 5000
            query_limit = query_limit - nquery
            distortion = None
            stopping = 0.0001
            auto_terminate=False
            if self.dataset == 'cifar10':
                delta = 1e-2
                len_T = 2
            elif self.dataset == 'imagenet':
                delta = 1
                len_T = 4
            k=200
            module_3 = OPT_attack_sign_SGD(self.model,k,delta,len_T,self.testset)
            
            timg = adv
            if self.targeted:
                adv, nqry, Dt = module_3.attack_targeted(oimg, olabel, timg, tlabel, alpha, beta, iterations, query_limit, distortion, self.seed, stopping, auto_terminate)
            else:
                adv, nqry, Dt = module_3.attack_untargeted(oimg, olabel, timg ,alpha, beta, iterations, query_limit, distortion,self.seed, stopping, auto_terminate)
                    
            print('Module 3: Finished SignOPT')

            D[nquery:nquery + nqry] = Dt

        return adv, nqry, D