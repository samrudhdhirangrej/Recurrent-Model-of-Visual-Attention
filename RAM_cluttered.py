from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from torch.distributions.normal import Normal
from RMVA import MODEL, LOSS, adjust_learning_rate
from random import shuffle
                
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if device.type=='cuda' else {}
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True,
                                           transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])),
                                           batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False,
                                           transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])),
                                           batch_size=batch_size, shuffle=True, **kwargs)

T = 4
lr = 0.0001
std = 0.25
scale = 3
decay = 0.975
im_sz = 60
glimpse_width = 12
model = MODEL(im_sz=im_sz, channel=1, glimps_width=glimpse_width, scale=scale, std = std).to(device)
loss_fn = LOSS(T=T, gamma=1, device=device).to(device)
optimizer = optim.Adam(list(model.parameters())+list(loss_fn.parameters()), lr=lr)


def add_clutter_and_translate_img(x, N_clutter, clutter_sz, to_sz):
    B,C,H,W = x.size()
    clutter_patches = []
    ind = H-clutter_sz+1

    for i in range(B):
        for _ in range(N_clutter):
            [r,c] = np.random.randint(0,ind,2)
            clutter_patches += [x[i,:,r:r+clutter_sz,c:c+clutter_sz]]
    shuffle(clutter_patches)
    x_t = -torch.ones(B,C,to_sz,to_sz).to(device) # background of MNIST is mapped to -1

    ind = to_sz-H+1
    ind_ = to_sz-clutter_sz+1
    for i in range(B):
        [loch, locw] = np.random.randint(0,ind,2)
        x_t[i,:,loch:loch+H,locw:locw+W] = x[i]
        for _ in range(N_clutter):
            [r,c] = np.random.randint(0,ind_,2)
            x_t[i,:,r:r+clutter_sz,c:c+clutter_sz] = torch.max(x_t[i,:,r:r+clutter_sz,c:c+clutter_sz], clutter_patches.pop())

    return x_t


for epoch in range(1,201):
    '''
    Training
    '''
    adjust_learning_rate(optimizer, epoch, lr, decay)
    model.train()
    train_aloss, train_lloss, train_bloss, train_reward = 0, 0, 0, 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = add_clutter_and_translate_img(data.to(device), N_clutter=4, clutter_sz=8, to_sz=im_sz)
        label = label.to(device) 
        optimizer.zero_grad()
        model.initialize(data.size(0), device)                   
        loss_fn.initialize(data.size(0))
        for _ in range(T):
            logpi, action = model(data)
            aloss, lloss, bloss, reward = loss_fn(action, label, logpi)  # loss_fn stores logpi during intermediate time-stamps and returns loss in the last time-stamp
        loss = aloss+lloss+bloss  
        loss.backward()
        optimizer.step()
        train_aloss += aloss.item()
        train_lloss += lloss.item()
        train_bloss += bloss.item()
        train_reward += reward.item()


    print('====> Epoch: {} Average loss: a {:.4f} l {:.4f} b {:.4f} Reward: {:.1f}'.format(
          epoch, train_aloss / len(train_loader.dataset),
          train_lloss / len(train_loader.dataset), 
          train_bloss / len(train_loader.dataset),
          train_reward *100/ len(train_loader.dataset)))


    # uncomment below line to save the model
    # torch.save([model.state_dict(), loss_fn.state_dict(), optimizer.state_dict()],'results/final'+str(epoch)+'.pth')

    '''
    Evaluation
    '''
    model.eval()
    test_aloss, test_lloss, test_bloss, test_reward = 0, 0, 0, 0
    for batch_idx, (data, label) in enumerate(test_loader):
        data = add_clutter_and_translate_img(data.to(device), N_clutter=4, clutter_sz=8, to_sz=im_sz)
        label = label.to(device) 
        model.initialize(data.size(0), device)
        loss_fn.initialize(data.size(0))
        for _ in range(T):
            logpi, action = model(data)
            aloss, lloss, bloss, reward = loss_fn(action, label, logpi)
        loss = aloss+lloss+bloss
        test_aloss += aloss.item()
        test_lloss += lloss.item()
        test_bloss += bloss.item()
        test_reward += reward.item()


    print('====> Epoch: {} Average loss: a {:.4f} l {:.4f} b {:.4f} Reward: {:.1f}'.format(
          epoch, test_aloss / len(test_loader.dataset),
          test_lloss / len(test_loader.dataset), 
          test_bloss / len(test_loader.dataset),
          test_reward *100/ len(test_loader.dataset)))


