import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from runx.logx import logx

from foolbox.distances import l0, l1, l2, linf
import math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score 

from art.attacks.evasion import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.utils import compute_success

import QEBA
from QEBA.criteria import TargetClass, Misclassification
from QEBA.pre_process.attack_setting import load_pgen
import random
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
       # print(f"q_eval: {q_eval}")
       # print(f"q_target: {q_target}")
       # print(f"loss: {loss}")
              
     
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()
        
def prediction(x):
    x_list = x[0].tolist()
    x_sort = sorted(x_list)
    max_index = x_list.index(x_sort[-1])

    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum

    return softmax, max_index#, sec_index

def AdversaryOne_Feature(args, shadowmodel, data_loader, cluster, Statistic_Data):
    Loss = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.cuda()
            output = shadowmodel(data)
            Loss.append(F.cross_entropy(output, target.cuda()).item())
    Loss = np.asarray(Loss)
    half = int(len(Loss)/2)
    member = Loss[:half]
    non_member = Loss[half:]        
    for loss in member:
        Statistic_Data.append({'DataSize':float(cluster), 'Loss':loss,  'Status':'Member'})
    for loss in non_member:
        Statistic_Data.append({'DataSize':float(cluster), 'Loss':loss,  'Status':'Non-member'})
    return Statistic_Data


# def AdversaryOne_evaluation(args, targetmodel, shadowmodel, data_loader, cluster, AUC_Loss, AUC_Entropy, AUC_Maximum):
#     Loss = []
#     Entropy = []
#     Maximum = []
#     with torch.no_grad():
#         for data, target in data_loader:
#             data, target = data.cuda(), target.cuda()
#             Toutput = targetmodel(data)
#             Tlabel = Toutput.max(1)[1]

#             Soutput = shadowmodel(data)
#             if Tlabel != target:
               
#                 Loss.append(100)
#             else:
#                 Loss.append(F.cross_entropy(Soutput, target).item())
            
#             prob = F.softmax(Soutput, dim=1) 

#             Maximum.append(torch.max(prob).item())
#             entropy = -1 * torch.sum(torch.mul(prob, torch.log(prob)))
#             if str(entropy.item()) == 'nan':
#                 Entropy.append(1e-100)
#             else:
#                 Entropy.append(entropy.item())
 
#     mem_groundtruth = np.ones(int(len(data_loader.dataset)/2))
#     non_groundtruth = np.zeros(int(len(data_loader.dataset)/2))
#     groundtruth = np.concatenate((mem_groundtruth, non_groundtruth))

#     dqn=DQN() 
    
#     predictions_Loss = np.asarray(Loss)
#     predictions_Entropy = np.asarray(Entropy)
#     predictions_Maximum = np.asarray(Maximum)
    
#     fpr, tpr, _ = roc_curve(groundtruth, predictions_Loss, pos_label=0, drop_intermediate=False)
#     AUC_Loss.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})

#     fpr, tpr, _ = roc_curve(groundtruth, predictions_Entropy, pos_label=0, drop_intermediate=False)
#     AUC_Entropy.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})

#     fpr, tpr, _ = roc_curve(groundtruth, predictions_Maximum, pos_label=1, drop_intermediate=False)
#     AUC_Maximum.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})
#     return AUC_Loss, AUC_Entropy, AUC_Maximum

# def AdversaryOne_evaluation(args, targetmodel, shadowmodel, data_loader, cluster, AUC_Loss, AUC_Entropy, AUC_Maximum):
#     Loss = []
#     Entropy = []
#     Maximum = []
#     num_members = 0
#     num_non_members = 0
    
#     with torch.no_grad():
#         for data, target in data_loader:
#             data, target = data.cuda(), target.cuda()
#             Toutput = targetmodel(data)
#             Tlabel = Toutput.max(1)[1]

#             Soutput = shadowmodel(data)
#             if Tlabel != target:
#                 num_non_members += 1
#                 Loss.append(100)
#             else:
#                 num_members += 1
#                 Loss.append(F.cross_entropy(Soutput, target).item())
            
#             prob = F.softmax(Soutput, dim=1) 

#             Maximum.append(torch.max(prob).item())
#             entropy = -1 * torch.sum(torch.mul(prob, torch.log(prob)))
#             if str(entropy.item()) == 'nan':
#                 Entropy.append(1e-100)
#             else:
#                 Entropy.append(entropy.item())

#     print(f"Number of members: {num_members}")
#     print(f"Number of non-members: {num_non_members}")
    
#     mem_groundtruth = np.ones(num_members)
#     non_groundtruth = np.zeros(num_non_members)
#     groundtruth = np.concatenate((mem_groundtruth, non_groundtruth))

#     dqn=DQN() 
    
#     predictions_Loss = np.asarray(Loss)
#     predictions_Entropy = np.asarray(Entropy)
#     predictions_Maximum = np.asarray(Maximum)
    
#     fpr, tpr, _ = roc_curve(groundtruth, predictions_Loss, pos_label=0, drop_intermediate=False)
#     AUC_Loss.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})

#     fpr, tpr, _ = roc_curve(groundtruth, predictions_Entropy, pos_label=0, drop_intermediate=False)
#     AUC_Entropy.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})

#     fpr, tpr, _ = roc_curve(groundtruth, predictions_Maximum, pos_label=1, drop_intermediate=False)
#     AUC_Maximum.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})
    
#     return AUC_Loss, AUC_Entropy, AUC_Maximum

def AdversaryOne_evaluation(args, targetmodel, shadowmodel, data_loader, cluster, AUC_Loss, AUC_Entropy, AUC_Maximum):
    data_list = list(data_loader.dataset)
    Loss = []
    Entropy = []
    Maximum = []
    num_members = 0
    num_non_members = 0
    num_mem= min([cluster, 1000, 2000])
  
    mem_correct=0
    non_correct=0
    mem_set = data_list[:num_mem]
    non_set = data_list[:-num_mem]
    
    dqn=DQN() 
    state_list=[]
    for i in range(20):    
        state_list.append(0)
        
    action_list=[]
    reward_list=[]
    j=0
    query_budget=0
    success=0
    success_list=[]
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            Toutput = targetmodel(data)  #     Toutput is the predicted label from target model
            #Tlabel = Toutput.max(1)[1]
            Tlabel = Toutput.max(1)[1].detach() # Tlabel is the predicted label from target model
            query_budget+=1
            
            if j<=20:
                action_list.append(0)
            else:
                if dqn.choose_action(np.array(state_list))==1:
                    random_int = random.choice([x for x in range(10) if x!= int(Tlabel)])
                    tensor_int = torch.tensor(random_int)
                    Tlabel = tensor_int.cuda()
                    action_list.append(1)   
                else:
                    action_list.append(0)      
                       
            Soutput = shadowmodel(data)  # Soutput is the predicted label from shadow model
            Slabel = Soutput.max(1)[1].detach()
            #print(f"Tlabel : {Tlabel}")
            #print(f"Slabel : {Slabel}")
            if torch.all(torch.eq(Tlabel, Slabel)):
                success+=1
                
            if Tlabel != target:  # target is the real label in the dataset
                num_non_members += 1
                Loss.append(100)
                if j >= (len(data_loader)-num_mem):
                    non_correct+=1  
            else:
                num_members += 1
                Loss.append(F.cross_entropy(Soutput, target).item())
                if j <= num_mem:
                    mem_correct+=1
            
            prob = F.softmax(Soutput, dim=1) 

            Maximum.append(torch.max(prob).item())
        
            entropy = -1 * torch.sum(torch.mul(prob, torch.log(prob)))
            if str(entropy.item()) == 'nan':
                Entropy.append(1e-100)
            else:
                Entropy.append(entropy.item())
                
            if action_list[j]==1:
                state_list.append(1)
                if j >= (len(data_loader)-num_mem):
                    reward_list.append(5)
                else:
                    reward_list.append(-5)
                
            else:
                state_list.append(0)
                if j >= (len(data_loader)-num_mem):
                    reward_list.append(-5)
                else:
                    reward_list.append(5)
 
            dqn.store_transition(state_list[0:20],action_list[j],reward_list[j],state_list[1:21])
            
            state_list.pop(0)
            
            #dqn.learn()
            if query_budget%200==0:
                print("The success rate is: {:.2f}".format(success/query_budget))
                print(query_budget)
                success_list.append(success/query_budget*0.5)
                
           
            j=j+1
    success_list.reverse()
    success_list = [round(elem, 2) for elem in success_list]
    print(success_list)
    #print(Maximum)
    #print(query_budget)
    print(f"Total number of members and no-members : {num_mem}")
    print(f"Number of members : {num_members}")
    print(f"Number of non-members: {num_non_members}")
    print(f"Number of right members : {mem_correct}")
    print(f"Number of right non_members : {non_correct}")
    
  
    mem_groundtruth = np.ones(num_members)
    non_groundtruth = np.zeros(num_non_members)
    groundtruth = np.concatenate((mem_groundtruth, non_groundtruth))

  
            
    predictions_Loss = np.asarray(Loss)
    predictions_Entropy = np.asarray(Entropy)
    predictions_Maximum = np.asarray(Maximum)
    
    fpr, tpr, _ = roc_curve(groundtruth, predictions_Loss, pos_label=0, drop_intermediate=False)
    AUC_Loss.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})

    fpr, tpr, _ = roc_curve(groundtruth, predictions_Entropy, pos_label=0, drop_intermediate=False)
    AUC_Entropy.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})

    fpr, tpr, _ = roc_curve(groundtruth, predictions_Maximum, pos_label=1, drop_intermediate=False)
    AUC_Maximum.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})
    
#     # Plot AUC changes with the increase of datasize
#     plt.plot([auc['DataSize'] for auc in AUC_Loss], [auc['AUC'] for auc in AUC_Loss], label="Loss")
#     plt.plot([auc['DataSize'] for auc in AUC_Entropy], [auc['AUC'] for auc in AUC_Entropy], label="Entropy")
#     plt.plot([auc['DataSize'] for auc in AUC_Maximum], [auc['AUC'] for auc in AUC_Maximum], label="Maximum")
#     plt.title("Changes of AUC with the Increase of Datasize")
#     plt.xlabel("Datasize")
#     plt.ylabel("AUC")
#     plt.legend()
#     plt.show()
    
    return AUC_Loss, AUC_Entropy, AUC_Maximum


def AdversaryTwo_HopSkipJump(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data=False, maxitr=50, max_eval=10000):
    input_shape = [(3, 32, 32), (3, 32, 32), (3, 64, 64), (3, 128, 128)]
    nb_classes = [10, 100, 43, 19]
    ARTclassifier = PyTorchClassifier(
                model=targetmodel,
                clip_values=(0, 1),
                loss=F.cross_entropy,
                input_shape=input_shape[args.dataset_ID],
                nb_classes=nb_classes[args.dataset_ID],
            )
    L0_dist, L1_dist, L2_dist, Linf_dist = [], [], [], []
    Attack = HopSkipJump(classifier=ARTclassifier, targeted =False, max_iter=maxitr, max_eval=max_eval)

    mid = int(len(data_loader.dataset)/2)
    member_groundtruth, non_member_groundtruth = [], []
    for idx, (data, target) in enumerate(data_loader): 
        targetmodel.module.query_num = 0
        data = np.array(data)  
        logit = ARTclassifier.predict(data)
        _, pred = prediction(logit)
        if pred != target.item() and not Random_Data:
            success = 1
            data_adv = data
        else:
            data_adv = Attack.generate(x=data) 
            data_adv = np.array(data_adv) 
            if Random_Data:
                success = compute_success(ARTclassifier, data, [pred], data_adv) 
            else:
                success = compute_success(ARTclassifier, data, [target.item()], data_adv)

        if success == 1:
            print(targetmodel.module.query_num)
            logx.msg('-------------Training DataSize: {} current img index:{}---------------'.format(cluster, idx))
            L0_dist.append(l0(data, data_adv))
            L1_dist.append(l1(data, data_adv))
            L2_dist.append(l2(data, data_adv))
            Linf_dist.append(linf(data, data_adv))

            if idx < mid:
                member_groundtruth.append(1)
            else:
                non_member_groundtruth.append(0)

        if Random_Data and len(L0_dist)==100:
            break
        
    member_groundtruth = np.array(member_groundtruth)
    non_member_groundtruth = np.array(non_member_groundtruth)

    groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))
    L0_dist = np.asarray(L0_dist)
    L1_dist = np.asarray(L1_dist)
    L2_dist = np.asarray(L2_dist)
    Linf_dist = np.asarray(Linf_dist)

    fpr, tpr, _ = roc_curve(groundtruth, L0_dist, pos_label=1, drop_intermediate=False)
    L0_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L1_dist, pos_label=1, drop_intermediate=False)
    L1_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L2_dist, pos_label=1, drop_intermediate=False)
    L2_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, Linf_dist, pos_label=1, drop_intermediate=False)
    Linf_auc = round(auc(fpr, tpr), 4)

    ### AUC based on distance
    auc_score = {'DataSize':float(cluster), 'L0_auc':L0_auc, 'L1_auc':L1_auc, 'L2_auc':L2_auc, 'Linf_auc':Linf_auc}
    AUC_Dist.append(auc_score)

    ### Distance of L0, L1, L2, Linf
    middle= int(len(L0_dist)/2)
    for idx, (l0_dist, l1_dist, l2_dist, linf_dist) in enumerate(zip(L0_dist, L1_dist, L2_dist, Linf_dist)):   
        if idx < middle:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Member'}
        else:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Non-member'}
        Distance.append(data)
    return AUC_Dist, Distance

def AdversaryTwo_QEBA(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data=False, max_iter=150):
    #input_shape = [(3, 32, 32), (3, 32, 32), (3, 64, 64), (3, 128, 128), (3, 64, 64)]
    nb_classes = [10, 100, 43, 19, 200]
    PGEN = ['resize768']
    p_gen, maxN, initN = load_pgen(args, PGEN[0])
    
    fmodel = QEBA.models.PyTorchModel(targetmodel, bounds=(0, 1), 
                num_classes=nb_classes[args.dataset_ID], discretize=False)
    Attack = QEBA.attacks.BAPP_custom(fmodel, criterion=Misclassification()) #criterion=TargetClass(src_label)
    L0_dist, L1_dist, L2_dist, Linf_dist = [], [], [], []
    member_groundtruth, non_member_groundtruth = [], []
    mid = int(len(data_loader.dataset)/2)
    for idx, (data, target) in enumerate(data_loader):   
        targetmodel.module.query_num = 0
        data = data.numpy()
        data = np.squeeze(data)
        pred = np.argmax(fmodel.forward_one(data))
        if pred != target.item():
            data_adv = data
            pred_adv = pred
        else:
            grad_gt = fmodel.gradient_one(data, label=target.item())
            rho = p_gen.calc_rho(grad_gt, data).item()

            Adversarial = Attack(data, label=target.item(), starting_point = None, iterations=max_iter, stepsize_search='geometric_progression', 
                        unpack=False, max_num_evals=maxN, initial_num_evals=initN, internal_dtype=np.float32, 
                        rv_generator = p_gen, atk_level=999, mask=None, batch_size=1, rho_ref = rho, 
                        log_every_n_steps=1, suffix=PGEN[0], verbose=False)  

        
            data_adv = Adversarial.perturbed     
            pred_adv = Adversarial.adversarial_class

        if target.item() != pred_adv and type(data_adv) == np.ndarray:
            print(targetmodel.module.query_num)
            logx.msg('-------------Training DataSize: {} current img index:{}---------------'.format(cluster, idx))
            data = data[np.newaxis, :]
            data_adv = data_adv[np.newaxis, :]
            L0_dist.append(l0(data, data_adv))
            L1_dist.append(l1(data, data_adv))
            L2_dist.append(l2(data, data_adv))
            Linf_dist.append(linf(data, data_adv))
            if idx < mid:
                member_groundtruth.append(1)
            else:
                non_member_groundtruth.append(0)
        if Random_Data and len(L0_dist)==100:
            break
    member_groundtruth = np.array(member_groundtruth)
    non_member_groundtruth = np.array(non_member_groundtruth)

    groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))
    L0_dist = np.asarray(L0_dist)
    L1_dist = np.asarray(L1_dist)
    L2_dist = np.asarray(L2_dist)
    Linf_dist = np.asarray(Linf_dist)

    fpr, tpr, _ = roc_curve(groundtruth, L0_dist, pos_label=1, drop_intermediate=False)
    L0_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L1_dist, pos_label=1, drop_intermediate=False)
    L1_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L2_dist, pos_label=1, drop_intermediate=False)
    L2_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, Linf_dist, pos_label=1, drop_intermediate=False)
    Linf_auc = round(auc(fpr, tpr), 4)

    ### AUC based on distance
    auc_score = {'DataSize':float(cluster), 'L0_auc':L0_auc, 'L1_auc':L1_auc, 'L2_auc':L2_auc, 'Linf_auc':Linf_auc}
    AUC_Dist.append(auc_score)

    ### Distance of L0, L1, L2, Linf
    middle= int(len(L0_dist)/2)
    for idx, (l0_dist, l1_dist, l2_dist, linf_dist) in enumerate(zip(L0_dist, L1_dist, L2_dist, Linf_dist)):   
        if idx < middle:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Member'}
        else:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Non-member'}
        Distance.append(data)
    return AUC_Dist, Distance
def AdversaryTwo_SaltandPepperNoise(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data=False, max_iter=150):
    nb_classes = [10, 100, 43, 19, 200]
    PGEN = ['resize768']
    # p_gen, maxN, initN = load_pgen(args, PGEN[0])
    
    fmodel = QEBA.models.PyTorchModel(targetmodel, bounds=(0, 1), 
                num_classes=nb_classes[args.dataset_ID], discretize=False)
    Attack = QEBA.attacks.SaltAndPepperNoiseAttack(fmodel, criterion=Misclassification()) #criterion=TargetClass(src_label)
    L0_dist, L1_dist, L2_dist, Linf_dist = [], [], [], []
    member_groundtruth, non_member_groundtruth = [], []
    mid = int(len(data_loader.dataset)/2)
    for idx, (data, target) in enumerate(data_loader):   
        targetmodel.module.query_num = 0
        data = data.numpy()
        data = np.squeeze(data)
        pred = np.argmax(fmodel.forward_one(data))
   
        if pred != target.item():
            data_adv = data
            pred_adv = pred
        else:

            data_adv = Attack(data, label=target.item())  

            if type(data_adv) == np.ndarray:
                pred_adv = np.argmax(fmodel.forward_one(data_adv))
            else:
                continue
        if target.item() != pred_adv:
            print(targetmodel.module.query_num)
            logx.msg('-------------Training DataSize: {} current img index:{}---------------'.format(cluster, idx))
            data = data[np.newaxis, :]
            data_adv = data_adv[np.newaxis, :]
            L0_dist.append(l0(data, data_adv))
            L1_dist.append(l1(data, data_adv))
            L2_dist.append(l2(data, data_adv))
            Linf_dist.append(linf(data, data_adv))
            if idx < mid:
                member_groundtruth.append(1)
            else:
                non_member_groundtruth.append(0)
        if Random_Data and len(L0_dist)==100:
            break
    member_groundtruth = np.array(member_groundtruth)
    non_member_groundtruth = np.array(non_member_groundtruth)

    groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))
    L0_dist = np.asarray(L0_dist)
    L1_dist = np.asarray(L1_dist)
    L2_dist = np.asarray(L2_dist)
    Linf_dist = np.asarray(Linf_dist)

    fpr, tpr, _ = roc_curve(groundtruth, L0_dist, pos_label=1, drop_intermediate=False)
    L0_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L1_dist, pos_label=1, drop_intermediate=False)
    L1_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L2_dist, pos_label=1, drop_intermediate=False)
    L2_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, Linf_dist, pos_label=1, drop_intermediate=False)
    Linf_auc = round(auc(fpr, tpr), 4)

    ### AUC based on distance
    auc_score = {'DataSize':float(cluster), 'L0_auc':L0_auc, 'L1_auc':L1_auc, 'L2_auc':L2_auc, 'Linf_auc':Linf_auc}
    AUC_Dist.append(auc_score)

    ### Distance of L0, L1, L2, Linf
    middle= int(len(L0_dist)/2)
    for idx, (l0_dist, l1_dist, l2_dist, linf_dist) in enumerate(zip(L0_dist, L1_dist, L2_dist, Linf_dist)):   
        if idx < middle:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Member'}
        else:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Non-member'}
        Distance.append(data)
    return AUC_Dist, Distance

