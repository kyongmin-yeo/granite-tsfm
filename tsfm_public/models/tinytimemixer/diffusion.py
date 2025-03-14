#
# Copyright (c) 2024 by Contributors for FMFastSim
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class Diffusion(nn.Module):
    def __init__(self, dim):
        self.dim_out = dim
        self.loc = 0
        self.scale = 1

    def set_diffusion(self,
                 diff_model='rescaled',       #diffusion model: 'orig' or 'rescaled'
                 dim_t_emb = 64,          #time embedding dimension
                 beta_info = {},        #noise schedule infor
                 uniform_t = True,
                 diff_substep = 15,     #training ratio
                 ):

        super().__init__()

        self.diff_model = diff_model
        self.uniform_t = uniform_t
        self.substep = diff_substep

        alpha,beta,gamma = noise_scheduler(**beta_info) 

        self.register_buffer('alpha',alpha.float())
        self.register_buffer('beta' ,beta .float())
        self.register_buffer('gamma',gamma.float())
        
        sigma = torch.zeros_like(beta)
        sigma[1:] = beta[1:]*(1-gamma[:-1])/(1-gamma[1:])
        self.register_buffer('sigma',sigma.sqrt().float())

        self.register_buffer('sqrt_one_minus_gamma',(1-gamma).sqrt().float())
        self.register_buffer('sqrt_gamma'          ,   gamma .sqrt().float())
        self.register_buffer('one_over_sqrt_alpha' ,(1/alpha).sqrt().float())

        if self.diff_model == 'orig':
            self.register_buffer('backward_coef_xt',torch.ones_like(beta.float()))
            self.register_buffer('backward_coef_ft',-(beta/(1-gamma).sqrt()).float())
        elif self.diff_model == 'rescaled':
            tmp = beta/(1-gamma)
            self.register_buffer('backward_coef_xt',(1-tmp).float())
            self.register_buffer('backward_coef_ft',   tmp .float())
        else:
            raise ValueError


        self.register_buffer('T',torch.tensor([beta.size(0)],dtype=torch.int32))

        self.dim_t = dim_t_emb

        self.sgd_counter = 0

        
    def get_parameter_projection(self, in_features: int):
        self.dim_in = in_features
        dim_t = self.dim_t

        dim_in  = self.dim_in
        dim_out = self.dim_out

        self.time_emb = nn.Sequential(Position_Embeddings(dim_t),
                                      nn.Linear(dim_t,  128),nn.SiLU(),
                                      nn.Linear(128  ,dim_t))

        self.cond_emb = nn.Sequential(nn.Linear(dim_in,  128),nn.SiLU(),
                                      nn.Linear(   128,dim_t))

        self.scale_emb = nn.Sequential(nn.Linear(2*dim_t,128),nn.SiLU(),
                                       nn.Linear(128,128),nn.SiLU(),
                                       nn.Linear(128,dim_out))

        self.pos_emb = nn.Sequential(nn.Linear(2*dim_t,128),nn.SiLU(),
                                     nn.Linear(128,128),nn.SiLU(),
                                     nn.Linear(128,dim_out))

        self.mean_pred = nn.Linear(dim_in,dim_out)

        self.fluc_pred = diff_net(dim_out,dim_out)

        return self.forward

    def prior_sampling(self,x0,t):

        drift = self.sqrt_gamma[t]*x0
        eps   = torch.randn_like(x0)

        xt = drift + self.sqrt_one_minus_gamma[t]*eps

        if self.diff_model == 'orig':
            return xt,eps
        elif self.diff_model == 'rescaled':
            return xt,drift
        else:
            raise ValueError

    def forward(self,x_in):
        self.mean_orig = self.mean_pred(x_in).transpose(-1,-2) #batch_size x prediction_length x nvar
        self.cond_var  = self.cond_emb(x_in.detach())
        return self.mean_orig

    @property
    def mean(self):
        return self.mean_orig*self.scale + self.loc

    def distribution(self,x_in,loc,scale):
        self.loc = loc
        self.scale = scale
        return self

    def one_step(self, x_in, t_in, c_in=None):
        nb = x_in.size(0)
        nv = x_in.size(1)

        if c_in == None:
            c0 = self.cond_var
        else:
            c0 = c_in
        t0 = self.time_emb(t_in)

        z_in = torch.cat([c0,t0],dim=-1)

        pos_emb   = self.  pos_emb(z_in)
        scale_emb = self.scale_emb(z_in)

        x_in = x_in*(scale_emb+1) + pos_emb

        x_out = self.fluc_pred(x_in)

        return x_out

    def sample(self,sample_size=(1,)):
        ns = sample_size[0]
        nb = self.mean.size(0)
        nv = self.mean.size(-1)

        y_mean = self.mean_orig.repeat_interleave(ns,dim=0)
        c_in   = self.cond_var .repeat_interleave(ns,dim=0)
        y_fluc = self.backward_sampling(y_mean.transpose(-1,-2),c_in=c_in)
        y_fluc = y_fluc.transpose(-1,-2)

        if torch.is_tensor(self.loc):
            scale = self.scale.repeat_interleave(ns,dim=0)
            loc   = self.loc  .repeat_interleave(ns,dim=0)
        else:
            scale = self.scale
            loc   = self.loc

        y_out = (y_mean+y_fluc)*scale + loc
        y_out = y_out.reshape(nb,ns,-1,nv).transpose(0,1) #num_samples x batch_size x prediction_length x num_channels
        return y_out

    def loss(self,target):
        nb = target.size( 0)
        nv = target.size(-1)

        #Mean component
        y_fluc = target - self.mean
        mean_loss = y_fluc.pow(2).mean()
        if self.sgd_counter%self.substep > 0:
            mean_loss = mean_loss.detach()

        #Fluctuating component
        #sample time
        if self.uniform_t:
            tt = torch.randint(self.T,(1,),device=target.device).expand(nb,nv,1)
        else:
            tt = torch.randint(self.T,(nb,),device=target.device).view(-1,1,1).repeat(1,nv,1)

        y_fluc = (y_fluc.detach()-self.loc)/self.scale
        y_fluc = y_fluc.transpose(-1,-2)

        xx,yy = self.prior_sampling(y_fluc,tt)
        y0 = self.one_step(xx,tt.squeeze(-1))
        
        fluc_loss = (y0-yy).mul(self.scale.transpose(-1,-2)).pow(2).mean()

        total_loss = mean_loss + fluc_loss

        self.sgd_counter += 1

        return total_loss

    @torch.no_grad()
    def backward_sampling(self,X,c_in=None):

        nb = X.size(0)
        nv = X.size(1)

        xT = torch.randn_like(X)

        x_new = xT
        for i in range(len(self.beta)):
            x_old = x_new

            tt = self.T.item()-i-1
            t_in = torch.tensor([tt],device=X.device).expand(nb,nv)

            drift = self.one_step(x_old,t_in,c_in=c_in)

            x_new = (self.backward_coef_xt[tt]*x_old + 
                     self.backward_coef_ft[tt]*drift)*self.one_over_sqrt_alpha[tt]
            x_new = x_new + self.sigma[tt]*torch.randn_like(x_old)

        return x_new

#define diffusion network
class diff_net(nn.Module):
    def __init__(self,dim_in,dim_out):
        super().__init__()
        self.dim = dim_out
        self.net = nn.ModuleList([nn.Linear(dim_in,dim_out),nn.Linear(dim_out,dim_out),nn.Linear(dim_out,dim_out)])
        self.final = nn.Linear(dim_out,dim_out)
    def forward(self,x_in):
        x0 = x_in
        for net in self.net:
            x1 = F.layer_norm(x0,(self.dim,))
            x0 = x0 + F.silu(net(x1))
        x_out = self.final(F.silu(x0))

        return x_out
        

#Positional Embedding
class Position_Embeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        self.register_buffer('embeddings',torch.exp(torch.arange(half_dim) * -embeddings))

    def forward(self, time):
        embeddings = time[:,:, None] * self.embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

#Define noise schedulers
def noise_scheduler(schedule='cos2',num_steps=200,tau=1,**kwargs):
    if schedule == 'cos':
        print('use cosine-1 gamma scheduler')
        gamma = cosine_schedule(num_steps,tau=tau)
    elif schedule == 'cos2':
        print('use cosine-2 gamma scheduler')
        gamma = cosine2_schedule(num_steps)
    elif schedule == 'sigmoid':
        print('use sigmoid gamma scheduler')
        gamma = sigmoid_schedule(num_steps,tau=tau)
    elif scheduler == 'linear':
        print('use linear gamma scheduler')
        gamma = linear_schedule(num_steps)
    else:
        print(f'{gamma_schedule} scheduler is not defined')
        raise ValueError

    alpha     = gamma*1.0
    alpha[1:] = alpha[1:]/gamma[:-1]
    beta      = 1-alpha

    print(f'beta : max {beta [-1].item()} and min {beta [ 0].item()}')
    print(f'gamma: max {gamma[ 0].item()} and min {gamma[-1].item()}')

    return alpha,beta,gamma


#define gamma scheduler
#number of steps: number of diffusion steps
def linear_schedule(num_steps=200):
    t = torch.arange(0,1+1.e-6,step=1/(num_steps+1),dtype=torch.double)

    gamma = 1-t
    gamma = gamma[1:-1] #remove the first and last knots
    return gamma

def sigmoid_schedule(num_steps,start=-3,end=3,tau=1):
    t = torch.arange(0,1+1.e-6,step=1/(num_steps+1),dtype=torch.double)

    x = ((t*(end-start)+start)/tau).sigmoid()

    gamma = (x-x[-1])/(x[0]-x[-1])
    gamma = gamma[1:-1] #remove the first and last knots
    return gamma

def cosine_schedule(num_steps,start=0,end=1,tau=1):
    t = torch.arange(0,1+1.e-6,step=1/(num_steps+1),dtype=torch.double)

    r = start/end
    x = ((t*(1-r)+r)*math.pi/2-1.e-6).cos().pow(2*tau)
    #x = ((t*(end-start)+start)*math.pi/2-1.e-6).cos().abs().pow(2*tau)
    #y = ((t*(end-start)+start)*math.pi/2-1.e-6).cos()

    gamma = (x-x[-1])/(x[0]-x[-1])
    gamma = gamma[1:-1] #remove the first and last knots
    return gamma

def cosine2_schedule(num_steps,s=0.008):
    t = torch.arange(0,1+1.e-6,step=1/(num_steps+1),dtype=torch.double)
    x = ((t+s)/(1+s)*math.pi*0.5).cos().pow(2)

    gamma = x/x[0]
    gamma = gamma[1:-1]
    return gamma
