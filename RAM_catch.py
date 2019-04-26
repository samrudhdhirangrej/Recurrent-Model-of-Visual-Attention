from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from catch import Catch

class RETINA(nn.Module):
    '''
        Retina is a bandlimited sensor.
        It extracts patches at given location at multiple scales.
        Patches are resized to smallest scale.
        Resized patches are stacked in channel dimension.
    '''
    def __init__(self, im_sz, width, scale):
        super(RETINA, self).__init__()

        self.hw = int(width/2)
        self.scale = int(scale)
        self.im_sz = im_sz

    def extract_patch_in_batch(self, x, l, scale):
        l = (self.im_sz*(l+1)/2).type('torch.IntTensor')
        low = l                                                           # lower boundaries of patches
        high = l + 2*(2**(scale-1))*self.hw                               # upper boundaries of patches
        patch = []
        for b in range(x.size(0)):
            patch += [x[b:b+1,:,low[b,0]:high[b,0], low[b,1]:high[b,1]]]  # extract patches
        return torch.cat(patch,0)

    def forward(self, x, l):
        B,C,H,W = x.size()
        padsz = (2**(self.scale-1))*self.hw
        x_pad = F.pad(x, (padsz, padsz, padsz, padsz), "replicate")       # pad image
        patch = self.extract_patch_in_batch(x_pad,l,self.scale)           # extract patch at highest scale

        # now we extract do the following for speed up:
        # 1. extract smaller scale patches from the center of the higher scale patches.
        # 2. resize (with maxpool) the extracted patches to the lowest scale.
        # 3. stack patches from all scales.

        out = [F.max_pool2d(patch, kernel_size=2**(self.scale-1))]        # step 2 and 3 for the highest scale
        cntr = int(patch.size(2)/2)
        halfsz = cntr
        for s in range(self.scale-1,0,-1):
            halfsz = int(halfsz/2)                                        # step 1,2 and 3 for other scales
            out += [F.max_pool2d(patch[:,:,cntr-halfsz:cntr+halfsz,cntr-halfsz:cntr+halfsz], kernel_size=2**(s-1))]
        out = torch.cat(out,1)

        return out


class GLIMPSE(nn.Module):

    ''''
    Glimpse network contains RETINA and an encoder.
    Encoder encodes output of RETINA and glimpse location.
    '''
    def __init__(self, im_sz, channel, glimps_width, scale):
        super(GLIMPSE, self).__init__()

        self.im_sz = im_sz
        self.ro    = RETINA(im_sz, glimps_width, scale)                   # ro(x,l)
        self.fc_ro = nn.Linear(scale * (glimps_width**2) * channel, 128)  # ro(x,l) -> hg
        self.fc_lc = nn.Linear(2, 128)                                    # l -> hl
        self.fc_hg = nn.Linear(128,256)                                   # f(hg)
        self.fc_hl = nn.Linear(128,256)                                   # f(hl)


    def forward(self, x, l):
        ro = self.ro(x, l).view(x.size(0),-1)        # ro = output of RETINA
        hg = F.relu(self.fc_ro(ro))                  # hg = fg(ro)
        hl = F.relu(self.fc_lc(l))                   # hl = fl(l)
        g  = F.relu(self.fc_hg(hg)+self.fc_hl(hl))   # g = fg(hg,hl)
        return g

class CORE(nn.Module):
    '''
    Core network is a recurrent network which maintains a behavior state.
    '''
    def __init__(self):
        super(CORE, self).__init__()

        self.core = nn.LSTMCell(input_size = 256, hidden_size = 256)
        
    def initialize(self, B):
        self.hidden = torch.zeros(B, 256).to(device)
        self.cell = torch.zeros(B, 256).to(device)

    def forward(self, g):
        self.hidden, self.cell = self.core(g, (self.hidden, self.cell))
        return self.hidden

class LOCATION(nn.Module):
    '''
    Location network learns policy for sensing locations.
    '''
    def __init__(self, std):
        super(LOCATION, self).__init__()

        self.std = std
        self.fc = nn.Linear(256,2)

    def forward(self, h):
        l_mu = self.fc(h)               # compute mean of Gaussian
        pi = Normal(l_mu, self.std)     # create a Gaussian distribution
        l = pi.sample()                 # sample from the Gaussian 
        logpi = pi.log_prob(l)          # compute log probability of the sample
        l = torch.tanh(l)               # squeeze location to ensure sensing within the boundaries of an image
        return logpi, l


class ACTION(nn.Module):
    '''
    Action network learn policy for task specific actions.
    Game of Catch has three task-specific actions: move paddle right, left or stay put.
    '''

    def __init__(self):
        super(ACTION, self).__init__()

        self.fc = nn.Linear(256,3)

    def forward(self, h):
        a_prob = torch.softmax(self.fc(h),1)  # compute probability of each action
        pi = Categorical(a_prob)              # create distribution
        a = pi.sample()                       # sample from the distribution
        logpi = pi.log_prob(a)                # compute log probability of the sample
        return logpi, a-1                     # a-1: 0,1,2 --> -1,0,1 (left, stay, right)

class MODEL(nn.Module):
    def __init__(self, im_sz, channel, glimps_width, scale, std):
        super(MODEL, self).__init__()

        self.glimps = GLIMPSE(im_sz, channel, glimps_width, scale)
        self.core   = CORE()
        self.location = LOCATION(std)
        self.action = ACTION()

    def initialize(self, B):
        self.core.initialize(B)                            # initialize states of core network
        self.l = (torch.rand((B,2))*2-1).to(device)        # start with a glimpse at random location

    def forward(self, x):
        g = self.glimps(x,self.l)                          # glimpse encoding
        state = self.core(g)                               # update states based on new glimpse
        logpi_l, self.l = self.location(state)             # predict location of next glimpse
        logpi_a, a = self.action(state)                    # predict task specific action
        return a, logpi_a, logpi_l

class LOSS(nn.Module):
    def __init__(self, gamma):
        super(LOSS, self).__init__()

        self.baseline = nn.Parameter(torch.zeros(1,1).to(device), requires_grad = True)
        self.gamma = gamma
        self.notinit = True

    def initialize(self, B):
        self.t = 0
        self.logpi_l = []
        self.logpi_a = []

    def forward(self, reward, logpi_a, logpi_l, done):
        self.logpi_l += [logpi_l]
        self.logpi_a += [logpi_a]
        if done:
            if self.notinit:                                 # data-dependant initialization
                self.baseline.data = reward.mean()
                self.notinit = False
            R = reward                                       # centered rewards
            a_loss, l_loss, b_loss = 0, 0, 0
            R_b = (R-self.baseline.detach())
            for logpi_l, logpi_a in zip(reversed(self.logpi_l), reversed(self.logpi_a)):
                a_loss += - (logpi_a * R_b).sum()            # REINFORCE
                l_loss += - (logpi_l.sum(-1) * R_b).sum()    # REINFORCE
                R_b = self.gamma * R_b                       # discounted centered rewards (although discount factor is always 1)
            b_loss = ((self.baseline - R)**2).sum()          # minimize SSE between reward and the baseline
            return a_loss, l_loss , b_loss
        else:
            return 0, 0, 0
                
def adjust_learning_rate(optimizer, epoch, lr):
    lr = max(lr * (0.99 ** epoch), 1e-7)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_batches = 350
im_sz = 24
glimps_width = 6
scale = 3
batch_size =  64
lr = 0.0001
std = 0.25

model = MODEL(im_sz=im_sz, channel=1, glimps_width=glimps_width, scale=scale, std = std).to(device)
loss_fn = LOSS(gamma=0).to(device)
optimizer = optim.Adam([{'params': model.parameters(), 'lr': lr}, {'params':loss_fn.parameters(), 'lr':lr}])
env = Catch(batch_size = batch_size, device = device)


for epoch in range(1, 701):
    adjust_learning_rate(optimizer, epoch, lr)
    model.train()
    train_aloss, train_lloss, train_bloss, train_reward = 0, 0, 0, 0
    for batch_idx in range(n_batches):
        optimizer.zero_grad()
        model.initialize(batch_size)
        loss_fn.initialize(batch_size)
        Done = 0
        while(not Done):
            data = env.getframe()                      # get frames of
            action, logpi_a, logpi_l = model(data)     # pass frames from the model to generate actions
            Done, reward = env.step(action)            # make actions and receive rewards
            aloss, lloss, bloss = loss_fn(reward, logpi_a, logpi_l, Done)
        loss = aloss+lloss+bloss                       # loss_fn stores logpi during intermediate time-stamps and returns loss in the last time-stamp
        loss.backward()
        optimizer.step()
        train_aloss += aloss.item()
        train_lloss += lloss.item()
        train_bloss += bloss.item()
        train_reward += reward.sum().item()


    print('====> Epoch: {} Average loss: a {:.4f} l {:.4f} b {:.4f} Reward: {:.1f} baseline: {:.3f}'.format(
         epoch, train_aloss / (n_batches*batch_size),
         train_lloss / (n_batches*batch_size), 
         train_bloss / (n_batches*batch_size),
         train_reward *100/ (n_batches*batch_size),
         loss_fn.baseline.mean().item()))

    # uncomment below line to save the model
    # torch.save([model.state_dict(), loss_fn.state_dict(), optimizer.state_dict()],'results/final'+str(epoch)+'.pth')

