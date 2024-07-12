from torchvision.utils import save_image
from torch.autograd import grad
import torch
import time
import random
import os, logging
import numpy as np

from brep_mi_utils import *
import shutil

action = -1

import torch.nn as nn
import torch.nn.functional as F

# Define the Net class (define the network)

N_ACTIONS = 10  # Adjusted for 10 possible labels in CIFAR-10
N_STATES = 3072 * 2  # Adjusted for concatenated image input

BATCH_SIZE = 32                                 # Number of samples
LR = 0.01                                       # Learning rate
EPSILON = 0.99                                  # Greedy policy
GAMMA = 0.9                                     # Reward discount
TARGET_REPLACE_ITER = 100                       # Frequency of target network update
MEMORY_CAPACITY = 2000                          # Memory capacity

class Net(nn.Module):
    def __init__(self):                         # Define a series of attributes for Net
        super(Net, self).__init__()             # Equivalent to nn.Module.__init__()

        self.fc1 = nn.Linear(N_STATES, 50)      # Set the first fully connected layer (input layer to hidden layer): from N_STATES neurons to 50 neurons
        self.fc1.weight.data.normal_(0, 0.1)    # Weight initialization (normal distribution with mean 0 and standard deviation 0.1)
        self.out = nn.Linear(50, N_ACTIONS)     # Set the second fully connected layer (hidden layer to output layer): from 50 neurons to N_ACTIONS neurons
        self.out.weight.data.normal_(0, 0.1)    # Weight initialization (normal distribution with mean 0 and standard deviation 0.1)

    def forward(self, x):                       # Define the forward function (x is the state)
        x = F.relu(self.fc1(x))                 # Connect the input layer to the hidden layer, and use the ReLU activation function to process the value after the hidden layer
        actions_value = self.out(x)             # Connect the hidden layer to the output layer, and get the final output value (i.e., action value)
        return actions_value 
    
class DQN(object):
    def __init__(self):                         # Define a series of attributes for DQN
        self.eval_net, self.target_net = Net(), Net()  # Create two neural networks using Net: evaluation network and target network
        self.learn_step_counter = 0             # For target updating
        self.memory_counter = 0                 # For storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # Initialize the memory, each row represents a transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # Use the Adam optimizer (input is the evaluation network parameters and learning rate)
        self.loss_func = nn.MSELoss()           # Use mean squared error loss function (loss(xi, yi) = (xi - yi)^2)

    def choose_action(self, x):                 # Define the action selection function (x is the state)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # Convert x to 32-bit floating point format and add a dimension of 1 at dim=0
        if np.random.uniform() < EPSILON:       # Generate a random number in [0, 1), if less than EPSILON, choose the optimal action
            actions_value = self.eval_net.forward(x)  # Get the action value by feeding the state x to the evaluation network
            action = torch.max(actions_value, 1)[1].data.numpy()  # Output the index of the maximum value in each row, and convert to numpy ndarray format
            action = action[0]                  # Output the first number of action
        else:                                   # Randomly choose an action
            action = np.random.randint(0, N_ACTIONS)  # Here action is randomly 0 to 9 (N_ACTIONS = 10)
        return action                           # Return the chosen action (0 to 9)

    def store_transition(self, s, a, r, s_):    # Define the memory storage function (here input is a transition)
        transition = np.hstack((s, [a, r], s_))  # Concatenate arrays horizontally
        # If the memory is full, overwrite old data
        index = self.memory_counter % MEMORY_CAPACITY  # Get the row number where the transition will be placed
        self.memory[index, :] = transition       # Place the transition
        self.memory_counter += 1                 # Increment memory_counter by 1

    def learn(self):                            # Define the learning function (start learning after the memory is full)
        # Target network parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # Trigger at the beginning, then every 100 steps
            self.target_net.load_state_dict(self.eval_net.state_dict())  # Assign the evaluation network parameters to the target network
        self.learn_step_counter += 1             # Increment learn_step_counter by 1

        # Sample a batch of data from the memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # Randomly sample 32 numbers from [0, 2000), may repeat
        b_memory = self.memory[sample_index, :]   # Extract 32 transitions corresponding to the 32 indices, store in b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # Extract 32 states, convert to 32-bit floating point format, and store in b_s, b_s has 32 rows and 4 columns
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        # Extract 32 actions, convert to 64-bit integer (signed) format, and store in b_a (it is LongTensor type to facilitate the use of torch.gather), b_a has 32 rows and 1 column
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        # Extract 32 rewards, convert to 32-bit floating point format, and store in b_r, b_r has 32 rows and 1 column
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # Extract 32 next states, convert to 32-bit floating point format, and store in b_s_, b_s_ has 32 rows and 4 columns

        # Get the evaluation and target values for 32 transitions, and update the evaluation network parameters using the loss function and optimizer
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s) outputs a series of action values for each b_s through the evaluation network, then .gather(1, b_a) aggregates the Q values of the corresponding indices b_a for each row
        q_next = self.target_net(b_s_).detach()
        # q_next does not propagate the error backward, so detach; q_next represents a series of action values for each b_s_ output by the target network
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0] means only returning the maximum value of each row, not the index (a one-dimensional tensor of length 32); .view() converts the previous one-dimensional tensor into the shape of (BATCH_SIZE, 1); finally get the target value through the formula
        loss = self.loss_func(q_eval, q_target)
        # Input 32 evaluation values and 32 target values, use mean squared error loss function
        self.optimizer.zero_grad()               # Clear the residual update parameter values of the previous step
        loss.backward()                          # Backpropagate the error, calculate the parameter update value
        self.optimizer.step()                    # Execute a single optimization step (parameter update)

# Sample "#points_count" points around a sphere centered on "current_point" with radius = "sphere_radius"
def gen_points_on_sphere(current_point, points_count, sphere_radius):
    # get random perturbations
    points_shape = (points_count,) + current_point.shape
    perturbation_direction = torch.randn(*points_shape).cuda()
    dims = tuple([i for i in range(1, len(points_shape))])
    
    # normalize them such that they are uniformly distributed on a sphere with the given radius
    perturbation_direction = (sphere_radius / torch.sqrt(torch.sum(perturbation_direction ** 2, axis=dims, keepdims=True))) * perturbation_direction
    
    # add the perturbations to the current point
    sphere_points = current_point + perturbation_direction
    return sphere_points, perturbation_direction

def attack_single_target(current_point, target_class, current_loss, G, target_model, evaluator_model, attack_params, criterion, current_iden_dir, dqn):
    current_iter = 0
    last_iter_when_radius_changed = 0
    
    # create log file
    log_file = open(os.path.join(current_iden_dir,'train_log'), 'w')
    losses = []
    target_class_tensor = torch.tensor([target_class]).cuda()
    current_sphere_radius = attack_params['current_sphere_radius']
    
    last_success_on_eval = False
    
    # Outer loop handles all sphere radii
    while current_iter - last_iter_when_radius_changed < attack_params['max_iters_at_radius_before_terminate']:
        
        # inner loop handles one single sphere radius
        while current_iter - last_iter_when_radius_changed < attack_params['max_iters_at_radius_before_terminate']:
            
            new_radius = False
            
            # step size is similar to learning rate
            # we limit max step size to 3. But feel free to change it
            step_size = min(current_sphere_radius / 3, 3)
            
            # sample points on the sphere
            new_points, perturbation_directions = gen_points_on_sphere(current_point, attack_params['sphere_points_count'], current_sphere_radius)
            
            # get the predicted labels of the target model on the sphere points
            new_points_classification = is_target_class(G(new_points), target_class, target_model)
            
            # handle case where all (or some percentage) sphere points lie in decision boundary. We increment sphere size
            if new_points_classification.sum() > 0.75 * attack_params['sphere_points_count']:
                save_tensor_images(G(current_point.unsqueeze(0))[0].detach(),
                                   os.path.join(current_iden_dir, "last_img_of_radius_{:.4f}_iter_{}.png".format(current_sphere_radius, current_iter)))
                # update the current sphere radius
                current_sphere_radius = current_sphere_radius * attack_params['sphere_expansion_coeff']
                
                log_file.write("new sphere radius at iter: {} ".format(current_iter))
                new_radius = True
                last_iter_when_radius_changed = current_iter
            
            # get the update direction, which is the mean of all points outside boundary if 'repulsion_only' is used. Otherwise it is the mean of all points * their classification (1, -1)
            if attack_params['repulsion_only'] == True:
                new_points_classification = (new_points_classification - 1) / 2
                
            grad_direction = torch.mean(new_points_classification.unsqueeze(1) * perturbation_directions, axis=0) / current_sphere_radius

            # move the current point with stepsize towards grad_direction
            current_point_new = current_point + step_size * grad_direction
            current_point_new = current_point_new.clamp(min=attack_params['point_clamp_min'], max=attack_params['point_clamp_max'])
            
            current_img = G(current_point_new.unsqueeze(0))
            img_tensor = current_img 
            img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

            # Convert tensor to data type torch.float32
            img_tensor = img_tensor.float()

            # Save tensor as image
            save_image(img_tensor, 'current_image.png', nrow=16, padding=2, normalize=False)

            if is_target_class(current_img, target_class, target_model)[0] == -1:
                log_file.write("current point is outside target class boundary")
                break

            current_point = current_point_new
            _, current_loss = decision(current_img, target_model, score=True, criterion=criterion, target=target_class_tensor)

            if current_iter % 50 == 0 or (current_iter < 200 and current_iter % 20 == 0):
                save_tensor_images(current_img[0].detach(), os.path.join(current_iden_dir, "iter{}.png".format(current_iter)))
            
            eval_decision = decision_Evaluator(current_img, evaluator_model)
            correct_on_eval = True if eval_decision == target_class else False
            if new_radius:
                point_before_inc_radius = current_point.clone()
                last_success_on_eval = correct_on_eval
                break
            
            iter_str = "iter: {}, current_sphere_radius: {}, step_size: {:.2f} sum decisions: {}, loss: {:.4f}, eval predicted class {}, classified correct on Eval {}".format(
                current_iter, current_sphere_radius, step_size,
                new_points_classification.sum(),
                current_loss.item(),
                eval_decision,
                correct_on_eval)
            
            log_file.write(iter_str + '\n')
            losses.append(current_loss.item())
            current_iter += 1

            # Use DQN to select a new target class
            state = np.concatenate((current_img.cpu().numpy().flatten(), current_point.cpu().numpy().flatten()))
            action = dqn.choose_action(state)
            new_label = action  # DQN selects the new label

            # Update the target class for the next iteration
            target_class = new_label
            target_class_tensor = torch.tensor([target_class]).cuda()

    log_file.close()
    acc = 1 if last_success_on_eval is True else 0
    return acc

def attack(attack_params, target_model, evaluator_model, generator_model, attack_imgs_dir, private_domain_imgs_path):
    # attack the same targets using same initial points as saved experiment
    if 'targets_from_exp' in attack_params:
        print("loading initial points from experiment dir: {}".format(attack_params['targets_from_exp']))
        points = gen_initial_points_from_exp(attack_params['targets_from_exp'])
        
    # attack same targets as experiment, but generate new random initial points    
    elif 'gen_idens_as_exp' in attack_params:
        print("attacking same targets as experiment dir: {}".format(attack_params['gen_idens_as_exp']))
        points = gen_idens_as_exp(attack_params['gen_idens_as_exp'],                                           
                                   attack_params['batch_dim_for_initial_points'],
                                   generator_model,
                                   target_model,
                                   attack_params['point_clamp_min'],
                                   attack_params['point_clamp_max'],
                                   attack_params['z_dim'])
    # attack target classes from iden_range_min to iden_range_max
    elif attack_params['targeted_attack']:
        print("attacking the targets from: {} to {}".format(attack_params['iden_range_min'], attack_params['iden_range_max']))
        points = gen_initial_points_targeted(attack_params['batch_dim_for_initial_points'],
                                             generator_model,
                                             target_model,
                                             attack_params['point_clamp_min'],
                                             attack_params['point_clamp_max'],
                                             attack_params['z_dim'],
                                             attack_params['iden_range_min'],
                                             attack_params['iden_range_max'])
    # attack any N labels
    else:
        print("attacking any {} targets".format(attack_params['num_targets']))
        points = gen_initial_points_untargeted(attack_params['num_targets'],
                                               attack_params['batch_dim_for_initial_points'],
                                               generator_model,
                                               target_model,
                                               attack_params['point_clamp_min'],
                                               attack_params['point_clamp_max'],
                                               attack_params['z_dim'])
    
    criterion = nn.CrossEntropyLoss().cuda()
    correct_on_eval = 0
    current_iter = 0

    dqn = DQN()  # Initialize the DQN agent

    for target_class in points:
        current_iter += 1
        current_point = points[target_class].cuda()
        print(" {}/{}: attacking iden {}".format(current_iter, len(points), target_class))
        target_class_tensor = torch.tensor([target_class]).cuda()

        # save the first generated image, and current point (z) to the iden_dir
        current_iden_dir = os.path.join(attack_imgs_dir, "iden_{}".format(target_class))
        os.makedirs(current_iden_dir, exist_ok=True)
        first_img = generator_model(current_point.unsqueeze(0))
        
        save_tensor_images(first_img[0].detach(), os.path.join(current_iden_dir, "original_first_point.png".format(current_iter)))
        np.save(os.path.join(current_iden_dir, 'initial_z_point'), current_point.cpu().detach().numpy())
        
        # copy the groundtruth images of the target to the attack dir
        # please put all groundtruth images in one single image called all.png
        # the path to the groundtruth image of label should be  "$dataset_dir/label/all.png"
        if len(private_domain_imgs_path) > 0:
            shutil.copy(os.path.join(private_domain_imgs_path, str(target_class), 'all.png'), os.path.join(current_iden_dir, 'groundtruth_imgs.png'))
        
        # first image should always be inside target class
        # assert is_target_class(first_img, target_class, target_model).item() == 1

        _, initial_loss = decision(generator_model(current_point.unsqueeze(0)), target_model, score=True, criterion=criterion, target=target_class_tensor)
        
        correct_on_eval += attack_single_target(current_point, target_class, initial_loss, generator_model, target_model, evaluator_model, attack_params, criterion, current_iden_dir, dqn)
        current_acc_on_eval = correct_on_eval / current_iter
        print("current acc on eval model: {:.2f}%".format(current_acc_on_eval * 100))
        
    total_acc_on_eval = correct_on_eval / len(points)
    print("total acc on eval model: {:.2f}%".format(total_acc_on_eval * 100))
