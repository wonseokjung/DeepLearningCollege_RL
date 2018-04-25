import gym
from gym import wrappers
import math, os, random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from collections import namedtuple
from collections import deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import ppaquette_gym_super_mario


checkpoint_dir = './checkpoints/'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
print('CUDA!!!!!',use_cuda)

env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
env = wrappers.Monitor(env, 'gym-results', force=True)

train = True
retrain = False



input_size = np.array([env.observation_space.shape[0], env.observation_space.shape[1], 15])
output_size = 13

dis = 0.9
REPLAY_MEMORY = 20000



def ddqn_replay_train(mainDQN, targetDQN, train_batch, l_rate):
    x_stack = np.empty(0).reshape(0, mainDQN.input_size[0]*mainDQN.input_size[1]*mainDQN.input_size[2])
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)
    action_stack = np.empty(0).reshape(0, 60)
    for state, action_seq, action_next_seq, action , reward, next_state, done in train_batch:
        Q = mainDQN(state, action_seq)
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * targetDQN(next_state, action_next_seq).max(1)[0]
            
            
        if state is None:
            print("None State, ", action, ", ", reward, ", ", next_state,", ", done)
        else:
            y_stack = np.vstack([y_stack, Q.cpu().data.numpy()])
            x_stack = np.vstack([x_stack, state.reshape(-1, mainDQN.input_size[0]*mainDQN.input_size[1]*mainDQN.input_size[2])])
            action_stack = np.vstack([action_stack, np.reshape(action_seq, (-1, 60))])
    
    Qpred = mainDQN(x_stack, action_stack)
    loss = F.mse_loss(Qpred, Variable(torch.Tensor(y_stack)).type(FloatTensor))
    optm = optim.Adam(mainDQN.parameters())
    optm.zero_grad()
    loss.backward()
    optm.step()
    
    return loss


def bot_play(mainDQN, env=env):
    start = env.reset()
    reward_sum = 0
    while True:
        if state in None or state.size ==1:
            output = random.randint(0, output_size-1)
            action = OutputToAction3(output)
            print("random action:", output)
        else:
            output = np.argmax(mainDQN(state))
            action = OutputToAction3(output)
            print("predicted action:", output)
        for n in range(len(action)):
            state, reward, done, info = env.step(action[n])
            if done:
                break
        reward_sum += reward
        if done:
            print("Total score", reward_sum)
            break
            
def OutputToAction3(output):
    actions={ # A: jump , B: run
        '0' :[[0,0,0,0,0,0], 'Nope'],
        '1' :[[1,0,0,0,0,0], 'Up'],
        '2' :[[0,0,1,0,0,0], 'Down'],
        '3' :[[0,1,0,0,0,0], 'Left'],
        '4' :[[0,1,0,0,1,0], 'Left + A'],
        '5' :[[0,1,0,0,0,1], 'Left + B'],
        '6' :[[0,1,0,0,1,1], 'Left + A + B'],
        '7' :[[0,0,0,1,0,0], 'Right'],
        '8' :[[0,0,0,1,1,0], 'Right + A'],
        '9' :[[0,0,0,1,0,1], 'Right + B'],
        '10':[[0,0,0,1,1,1], 'Right + A + B'],
        '11':[[0,0,0,0,1,0], 'A'],
        '12':[[0,0,0,0,1,1], 'A + B']
    }
    return [np.array([actions[str(output)][0]]*2), actions[str(output)][1]]
    





class DQN(nn.Module):
    def __init__ (self, input_size, output_size):
        super(DQN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.model1 = nn.Sequential(
            nn.Conv2d(15,64,3,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,2),
            nn.ReLU(),
            
            nn.Conv2d(64,128,3,1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,2),
            
            nn.Conv2d(128,256,3,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1),
            nn.ReLU(), 
            nn.Conv2d(256,256,3,2),
            
            nn.Conv2d(256,512,3,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,2),
            
            nn.Conv2d(512,512,3,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,2),
            
            nn.Conv2d(512,100,[2,3],1),
            nn.ReLU()
            
        )
        self.model2 = nn.Sequential(
            nn.Linear(160,160),
            nn.ReLU(),
            nn.Linear(160,50),
            nn.ReLU(),
            nn.Linear(50,self.output_size)
        )
    def forward(self, x, y):
        #x = np.ascontiguousarray(x, dtype=np.float32)
        x = np.reshape(x, [-1, 224,256,15])
        x = np.transpose(x, (0,3,1,2))
        x = Variable(torch.Tensor(x)).type(FloatTensor)
        y = Variable(torch.from_numpy(np.array(y))).type(FloatTensor)
        x = self.model1(x)
        x = x.view(x.size(0), 100)
        y = y.view(-1, 60)
        z = torch.cat((x,y),1)
        return self.model2(z)
    
   



def main():
    if train:
        init_episode = 1
        max_episode = 10000
        replay_buffer = deque()
        state_buffer = deque()
        next_state_buffer = deque()
        output_buffer = deque()
        
        mainDQN = DQN(input_size, output_size)
        targetDQN = DQN(input_size, output_size)
        targetDQN.load_state_dict(mainDQN.state_dict())
        
        
        if retrain:
            mainDQN = torch.load(mainDQN, 'checkpoints/mainDQN.pt')
            
        if use_cuda:
            mainDQN.cuda()
            targetDQN.cuda()
        
        for episode in range(init_episode, max_episode):
            e = 1.0 / (episode/500 + 1)
            print ("episode:", episode, ", epsilon: ", e)
            done = False
            step_count = 0
            state = env.reset()
            score = 0
            distance = 0
            prev_output = -1
            repeat = 0
            
            while not done:
                if np.random.rand(1) < e or state is None or state.size == 1 or step_count <=10:
                    output = random.randint(0, output_size -1)
                    action,action_name = OutputToAction3(output)
                    print("random action:", action_name)
                
                else:
                    predicted = mainDQN(acc_state, output_seq)
                    output = int(predicted.max(1)[1].view(1,1))
                    action,action_name = OutputToAction3(output)
                    print("output:",action_name, "predicted:", predicted)
                    
                for n in range(len(action)):
                    next_state, reward, done, info = env.step(action[n])
                    if done:
                        print('%dth:' %n)
                        break
                
                print('reward: ', reward)
                state_buffer.append(next_state)
                output_buffer.append(action)
                
                prev_distance = distance
                distance = info['distance']
                got_distance = distance - prev_distance
                
                past_score = score
                score = info['score']
                got_score = score - past_score
                
                time = info['time']
                
                reward = got_score/50 + got_distance/30
                
                if reward>0:
                    print("reward:", reward)
                if done:
                    reward -= 1.0
                    if distance>=3000:
                        reward = 1
                    print("last reward: ", reward)
                
                if step_count>=10:
                    acc_state = [state_buffer[-2-k] for k in range(5)]
                    state_buffer.popleft()
                    acc_state = np.reshape(acc_state, input_size[:3])
                    acc_next_state = [state_buffer[-1-k] for k in range(5)]
                    acc_next_state = np.reshape(acc_next_state, input_size[:3])
                    
                    output_seq = [output_buffer[-2-k] for k in range(5)]
                    output_next_seq = [output_buffer[-1-k] for k in range(5)]
                    output_buffer.popleft()
                    
                    replay_buffer.append((acc_state, output_seq, output_next_seq, output, reward, acc_next_state, done))
                    
                    if replay_buffer[-1][6]:
                        for k in range(1, 5):
                            replay_buffer[-1-k] = tuple(
                                replay_buffer[-1-k][0:4] + (-pow(0.9,k),) + replay_buffer[-1-k][5:]) # ??????
                    if replay_buffer[-1][4] > 2.0 and replay_buffer[-1][6] == False:
                        for k in range(1, 5):
                            replay_buffer[-1-k] = tuple(
                                replay_buffer[-1-k][0:4] + (pow(0.9,k),) + replay_buffer[-1-k][5:])
                    
                    if len(replay_buffer) > REPLAY_MEMORY:
                        replay_buffer.popleft()
                    acc_state = acc_next_state
                    
                state = next_state
                step_count += 1
            if step_count > 100000:
                break

            if (episode+1) % 1 == 0:
                for _ in range(50):
                    if len(replay_buffer) >= 10:
                        sample_idx = random.sample(range(0, len(replay_buffer)), 10)
                        minibatch = []
                        for i in sample_idx:
                            minibatch.append(replay_buffer[i])

                        l_rate = (1e-5 - 1e-4)*(1/max_episode)*episode + 1e-4
                        loss = ddqn_replay_train(mainDQN, targetDQN, minibatch, l_rate=l_rate)

                        print("Loss: %.3f,  l_rate: %.6f" %(loss, l_rate))

            if (episode+1) % 2 == 0:
                targetDQN.load_state_dict(mainDQN.state_dict())
                print('weights copied')

            if (episode+1) % 100 == 0:

                torch.save(mainDQN, 'checkpoints/mainDQN.pt')
            
            #env2 = wrappers.Monitor(env, 'gym-results', force=True)
            #for i in range(200):
            #    bot_play(mainDQN, env=env2)
            #env2.close()
    else:
        mainDQN = DQN(input_size, output_size)
        targetDQN = DQN(input_size, output_size)
        for i in range(200):
            bot_play(mainDQN, env=env)
        env.close()
        
if __name__ == "__main__":
    main()
        
                        
                    





