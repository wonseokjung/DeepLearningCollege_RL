import gym
import ppaquette_gym_super_mario 
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import pylab

from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

FILE_NAME = "dueling_ddqn_mario"
MAPPING_ACTION = {
                    0: [0, 0, 0, 0, 0, 0], # None
                    1: [1, 0, 0, 0, 0, 0], # Up
                    2: [0, 0, 1, 0, 0, 0], # Down
                    3: [0, 1, 0, 0, 0, 0], # Left
                    4: [0, 1, 0, 0, 1, 0], # Left + A
                    5: [0, 1, 0, 0, 0, 1], # Left + B
                    6: [0, 1, 0, 0, 1, 1], # Left + A + B
                    7: [0, 0, 0, 1, 0, 0], # Right
                    8: [0, 0, 0, 1, 1, 0], # Right + A
                    9: [0, 0, 0, 1, 0, 1], # Right + B
                    10: [0, 0, 0, 1, 1, 1], # Right + A + B
                    11: [0, 0, 0, 0, 1, 0], # A
                    12: [0, 0, 0, 0, 0, 1], # B
                    13: [0, 0, 0, 0, 1, 1], # A + B
                }

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class MarioAction(gym.Wrapper):
    def __init__(self, env):
        super(MarioAction, self).__init__(env)
        self.action_space = gym.spaces.Discrete(14)
    
    def _action(self, action_num):
        return MAPPING_ACTION.get(action_num)

    def _reverse_action(self, action):
        for a in MAPPING_ACTION.keys():
            if action == MAPPING_ACTION[a]:
                return a

        return 0

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(observe):
        processed_observe = np.uint8(
            resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
        return processed_observe



class Model(nn.Module):
    def __init__(self, action_size):
        super(Model, self).__init__()
        self.action_size = action_size

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1).cuda()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1).cuda()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1).cuda()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda()

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1).cuda()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda()

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1).cuda()
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2).cuda()

        self.linear1 = nn.Linear(2048, 4096).cuda()
        self.linear2 = nn.Linear(4096, 4096).cuda()

        self.value = nn.Linear(4096, 1).cuda()
        self.advantage = nn.Linear(4096, self.action_size).cuda()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = F.relu((self.conv1(x))) 
        x = F.relu((self.conv2(x))) 
        x = self.pool(x)            

        x = F.relu((self.conv3(x))) 
        x = F.relu((self.conv4(x))) 
        x = self.pool(x)            

        x = F.relu((self.conv5(x))) 
        x = F.relu((self.conv6(x)))
        x = F.relu((self.conv6(x))) 
        x = self.pool(x)

        x = F.relu((self.conv7(x))) 
        x = F.relu((self.conv8(x))) 
        x = F.relu((self.conv8(x))) 
        x = self.pool(x)

        x = F.relu((self.conv8(x))) 
        x = F.relu((self.conv8(x))) 
        x = F.relu((self.conv8(x))) 
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        v = self.value(x)
        a = self.advantage(x)
        q = v + a - a.mean(1, keepdim=True)

        return q

#    def flatten_size(self, *size):
#        d = 1
#        for n in range(size[1:]):
#            d *= n
#        return x.view((size[0], d))
    

class DQNAgent(object):
    def __init__(self, action_size):
        self.render = False
        self.load_mode = False

        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # Epsilon
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        # Hyperparameter
        self.batch_size = 32
        self.train_start = 100
        self.update_target_rate = 1000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=40000)
        self.no_op_steps = 30
        # Model
        self.model = Model(self.action_size)
        self.model.apply(self.weights_init)
        self.target_model = Model(self.action_size)

        self.optimizer = optim.RMSprop(self.model.parameters())

        self.avg_loss = 0
        self.avg_max_q = 0

        if self.load_mode:
            self.model = torch.load("save_model/" + FILE_NAME + ".pt")
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, history):
        ran_num = random.random()

        if ran_num > self.epsilon:
            pred_q = self.model(Variable(history, volatile=False).cuda()).data
            return pred_q.max(1)[1].view(1,1)
        else:
            return LongTensor([[random.randrange(self.action_size)]])

    def append_sample(self, history, action, reward, next_history, done):
        self.memory.append([history, action,reward, next_history, done])

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)
        elif classname.find('Conv') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        minibatch = random.sample(self.memory, self.batch_size)
        history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))

        action = np.zeros((self.batch_size,1))
        reward = np.zeros((self.batch_size,1))
        done = np.zeros((self.batch_size,1))

        for i in range(self.batch_size):
            history[i] = minibatch[i][0] / 255.
            next_history[i] = minibatch[i][3] / 255.
            action[i] = minibatch[i][1]
            reward[i] = minibatch[i][2]
            done[i] = minibatch[i][4]

        next_history = Variable(torch.from_numpy(next_history).type(FloatTensor)).cuda()
        target_val = self.model(next_history).data
        target_val = self.target_model(next_history).data

        for i in range(self.batch_size):
            if done[i]:
                target[i] = reward[i]
            else:
                max_action = target_val[i].max(0)[1]
                target[i] = reward[i] + self.discount_factor * target_val[i][max_action]

        history = Variable(torch.from_numpy(history).type(FloatTensor)).cuda()
        action = Variable(torch.from_numpy(action).type(LongTensor)).cuda()
        reward = Variable(torch.from_numpy(reward)).cuda()
        done = Variable(torch.from_numpy(done)).cuda()
        
        target = Variable(torch.from_numpy(target).type(FloatTensor)).cuda()
        q_val = self.model(history).gather(1, action)
        loss = F.smooth_l1_loss(q_val, target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        self.avg_loss += loss.data[0]

def pre_processing(observe):
    processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

if __name__ == "__main__":
    env = gym.make('ppaquette/SuperMarioBros-1-1-v0') 
    env = MarioAction(env)
    env = ProcessFrame84(env)
    state_size = env.observation_space.shape[0]
    action_size = 14

    agent = DQNAgent(action_size=action_size)
    num_episodes = 5000
    scores, episodes, global_step = [], [], 0 

    for e in range(num_episodes):
        done = False
        step, score, start_life = 0, 0, 5
        observe = env.reset()

        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))
        history = torch.from_numpy(history).type(FloatTensor)

        while not done:
            if agent.render:
                env.render()

            global_step += 1
            step += 1

            action = agent.get_action(history)[0,0]
            actions = MAPPING_ACTION.get(action)

            ## Double action
            _, _, _, _ = env.step(actions)
            observe, reward, done, info = env.step(actions)

            next_state = pre_processing(observe)
            next_state = np.reshape(next_state, (1, 84, 84, 1))
            next_history = np.append(history[:, :, :, :3], next_state, axis=3)
            next_history = torch.from_numpy(next_history)
            score += reward

            agent.avg_max_q += agent.model(Variable(history/255.).cuda()).data.max(1)[0]

            agent.append_sample(history, action, reward, next_history, done)
            
            if len(agent.memory) > agent.train_start:
                agent.train_model()

            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()


            history = next_history

            if done:
                scores.append(score)
                episodes.append(e)

                avg_loss = agent.avg_loss / float(step)
                avg_max_q = float(agent.avg_max_q / float(step))

                print("episode:", e, "  score:", score, "  memory length:",
                       len(agent.memory), "  epsilon:", agent.epsilon,
                       "  global_step:", global_step, "  average_q:",
                       avg_max_q, "  average loss:",
                       avg_loss)
                
                agent.avg_max_q, agent.avg_loss = 0, 0

                if e % 2 == 0:
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("./save_graph/" + FILE_NAME + ".png")

        if e % 1000 == 0:
            torch.save(agent.model, "save_model/" + FILE_NAME + ".pt")

    pylab.plot(episodes, scores, 'b')
    pylab.savefig("./save_graph/" + FILE_NAME + ".png")

