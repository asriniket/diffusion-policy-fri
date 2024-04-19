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

def get_dataloader(demonstrations, obs_horizon, pred_horizon, pose_stats, orientation_stats, 
                   img_stats=None, batch_size=64):
    '''
    Gets the dataloader for the training set
    '''
    diff_dataset = dataset.DiffDataset(demonstrations, obs_horizon, pred_horizon, pose_stats, 
                               orientation_stats, img_stats)
    return torch.utils.DataLoader(diff_dataset, batch_size=batch_size, shuffle=True)