import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

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
        # 2. resize the extracted patches to the lowest scale.
        # 3. stack patches from all scales.
       
        out = [F.interpolate(patch, size=2*self.hw, mode='bilinear', align_corners = True)]   # step 2 and 3 for the highest scale
        cntr = int(patch.size(2)/2)
        halfsz = cntr
        for s in range(self.scale-1):
            halfsz = int(halfsz/2)                                                            # step 1,2 and 3 for other scales
            out += [F.interpolate(patch[:,:,cntr-halfsz:cntr+halfsz,cntr-halfsz:cntr+halfsz], size=2*self.hw, mode='bilinear', align_corners = True)]
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

        self.fc_h = nn.Linear(256,256)
        self.fc_g = nn.Linear(256,256)

    def forward(self, h, g):
        return F.relu(self.fc_h(h) + self.fc_g(g)) # recurrent connection

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
    In case of classification actions are possible classes.
    This network will be trained with supervised loss in case of classification.
    '''
    def __init__(self):
        super(ACTION, self).__init__()

        self.fc = nn.Linear(256,10)

    def forward(self, h):
        return self.fc(h)  # Do not apply softmax as loss function will take care of it

class MODEL(nn.Module):
    '''
    Model combines all the previous elements
    '''
    def __init__(self, im_sz, channel, glimps_width, scale, std):
        super(MODEL, self).__init__()

        self.glimps = GLIMPSE(im_sz, channel, glimps_width, scale)
        self.core   = CORE()
        self.location = LOCATION(std)
        self.action = ACTION()

    def initialize(self, B, device):
        self.state = torch.zeros(B,256).to(device)    # initialize states of the core network
        self.l = (torch.rand((B,2))*2-1).to(device)   # start with a glimpse at random location

    def forward(self, x):
        g = self.glimps(x,self.l)                     # glimpse encoding
        self.state = self.core(self.state, g)         # update state of a core network based on new glimpse
        logpi, self.l = self.location(self.state)     # predict location of next glimpse
        a = self.action(self.state)                   # predict task specific actions
        return logpi, a

class LOSS(nn.Module):
    '''
    Loss function is tailored for the reward received at the end of the episode.
    Location network is trained with REINFORCE objective.
    Action network is trained with supervised objective.
    '''
    def __init__(self, T, gamma, device):
        super(LOSS, self).__init__()

        self.baseline = nn.Parameter(0.1*torch.ones(1,1).to(device), requires_grad = True) # initialize baseline to a reasonable value
        self.T = T                                                                         # length of an episode
        self.gamma = gamma                                                                 # discount factor

    def initialize(self, B):
        self.t = 0
        self.logpi = []

    def compute_reward(self, recon_a, a):
        return (torch.argmax(recon_a.detach(),1)==a).float() # reward is 1 if the classification is correct and zero otherwise
        

    def forward(self, recon_a, a, logpi):
        self.t += 1
        self.logpi += [logpi]
        if self.t==self.T:
            R = self.compute_reward(recon_a, a)                     # reward is given at the end of the episode
            a_loss = F.cross_entropy(recon_a, a, reduction='sum')   # supervised objective for action network 
            l_loss = 0
            R_b = (R - self.baseline.detach())                      # centered rewards
            for logpi in reversed(self.logpi):
                l_loss += - (logpi.sum(-1) * R_b).sum()             # REINFORCE
                R_b = self.gamma * R_b                              # discounted centered rewards (although discount factor is always 1)
            b_loss = ((self.baseline - R)**2).sum()                 # minimize SSE between reward and the baseline
            return a_loss , l_loss , b_loss, R.sum()
        else:
            return None, None, None, None
                
def adjust_learning_rate(optimizer, epoch, lr, decay_rate):
    '''
    Decay learning rate
    '''
    lr = lr * (decay_rate ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



