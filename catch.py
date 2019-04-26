from __future__ import print_function
import torch
import numpy as np
from matplotlib import pyplot as plt

class Catch():
    def __init__(self, grid_size=int(24), batch_size = 128, device = torch.device('cpu')):
        self.device = device
        self.gs = grid_size
        self.x_paddle = None                        # x-coordinate of paddle. There is no y-coordinate for the paddle 
        self.x_ball = None                          # x-coordinate for ball
        self.y_ball = None                          # y-coordinate for ball
        self.angle = None                           # angle of fall
        self.rad = lambda x: x * np.pi/180          # function to convert degree into randian
        self.B = batch_size

        self.reset(self.B)

    def reset(self, B):
        self.x_paddle = torch.randint(0,self.gs-1,(B,)).to(self.device)                # start at a random location
        self.x_ball   = torch.randint(0,self.gs,(B,)).float().to(self.device)          # start at a random location
        self.y_ball   = int(0)                                                         # ball is always in the first row of a canvas at the begining
        self.angle    = self.rad(torch.rand(B)*90 + 45).to(self.device)                # choose a random fall angle
        self.dx       = (torch.cos(self.angle)/np.cos(self.rad(45)))                   # compute howmuch ball moves in x-direction each timestamp

    def getframe(self):
        frame = torch.zeros(self.B, 1, self.gs, self.gs).to(self.device)               # create an empty canvas
        x_ball = self.x_ball.clamp(0,self.gs-1).long()                                 # valid positions for ball
        for i in range(self.B):
            frame[i,:,self.y_ball,x_ball[i]] = 1                                       # place ball
            frame[i,:,-1,self.x_paddle[i]:self.x_paddle[i]+2] = 1                      # place paddle
        return frame

    def step(self,action):
        self.y_ball += 1                                                                              # advance ball position
        self.x_ball = self.x_ball + self.dx

        out = ((self.x_ball<0) | (self.x_ball>=self.gs))                                              # check if ball is about to move out of the canvas
        self.dx = self.dx * torch.index_select(torch.Tensor([1,-1]).to(self.device), 0, out.long())   # if yes, bounce the ball in the canvas

        self.x_paddle = (self.x_paddle + action.long()).clamp(0,self.gs-2)                            # move paddle
        
        done = self.y_ball == self.gs-1                                                         
        reward = torch.zeros(self.B,).to(self.device)
        if done:
           x_ball = self.x_ball.clamp(0,self.gs-1).long()                                             # valid ball position
           reward = ((x_ball == self.x_paddle) | (x_ball == self.x_paddle+1)).float()                 # reward = 1 if ball is caught, else 0
           self.reset(self.B)                                                                         # reset when the episode ends

        return done, reward.detach()



