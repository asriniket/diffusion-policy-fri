# file imports
import dataset
import model

from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm


# Class to handle training model, saves important data
class Trainer():
    def __init__(self, demonstrations, obs_horizon, pred_horizon, 
                 pose_stats, orientation_stats, 
                 img_stats=None, batch_size=64, device="cpu", 
                 learning_rate=1e-4, save_file="checkpoint.pth"):
        '''
        Init function, saves relevant data in class.
        '''
        self.demos = demonstrations
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.pose_stats = pose_stats
        self.orientation_stats = orientation_stats
        self.img_stats = img_stats
        self.batch_size = batch_size
        self.device = device
        self.lr = learning_rate
        self.save_file = save_file


    def get_dataloader(self):
        '''
        Gets the dataloader for the training set.
        '''
        diff_dataset = dataset.DiffDataset(self.demos, self.obs_horizon, self.pred_horizon, self.pose_stats, 
                                self.orientation_stats, self.img_stats)
        return torch.utils.DataLoader(diff_dataset, batch_size=self.batch_size, shuffle=True)



    def get_model(self):
        '''
        Gets model and exponential moving average.
        '''
        vision_encoder = model.get_resnet("resnet18")
        vision_encoder = model.replace_bn_with_gn(vision_encoder)

        vision_feature_dim = 512
        lowdim_obs_dim = 7
        obs_dim = vision_feature_dim + lowdim_obs_dim
        action_dim = 7

        noise_pred_net = model.ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*self.obs_horizon
        )

        nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_pred_net
        })

        nets = nets.to(torch.device(self.device))

        ema = EMAModel(
            parameters=nets.parameters(),
            power=0.75
        )

        return nets, ema
    
    def get_noise_scheduler(self, num_diffusion_iters):
        '''
        Gets noise scheduler.
        '''
        return DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    
    def train(self, num_epochs=20, print_stats=True):
        '''
        Runs training loop.
        '''
        num_diffusion_iters = 100

        trainloader = self.get_dataloader()
        nets, ema = self.get_model()
        noise_scheduler = self.get_noise_scheduler(num_diffusion_iters)
