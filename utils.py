import os
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from collections import deque
from typing import Union

class Save_model_logfile:
    def __init__(self,
                 algo_name:str,
                 env_name="cstr_ode", 
                 seed=0):
        
        log_name = algo_name + '_logs'
        directory_name= algo_name + '_preTrained'
        
        if not os.path.exists(log_name):
            os.makedirs(log_name)

        log_name = log_name + '/' + env_name + '/'
        if not os.path.exists(log_name):
                os.makedirs(log_name)
                
        current_num_files = next(os.walk(log_name))[2]
        run_num = len(current_num_files)
        
        # path to log file
        self.log_f_name = log_name + '/' + algo_name + '_' + env_name + "_log_" + str(run_num) + ".csv"
        
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        directory_name = directory_name + '/' + env_name + '/'
        if not os.path.exists(directory_name):
                os.makedirs(directory_name)
        
        # path to model weighting      
        self.checkpoint_path = directory_name + algo_name + "_{}_{}_{}.pth".format(env_name, seed, run_num)
    
    def log_path(self):
        return self.log_f_name
    
    def model_path(self):
        return self.checkpoint_path
    
def get_grad_norm(model):
    with torch.no_grad():
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
    return total_norm 
 
class auto_clip:
    def __init__(self, history_size:Union[int, None]=None):
        if history_size is None:
            self.buffer = deque()
        else:
            self.buffer = deque(maxlen=history_size)
    
    @staticmethod
    def _get_grad_norm(model:nn.Module,norm_type:float):
        with torch.no_grad():
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        return total_norm 

    def _auto_clip_grad_norm(self, model:nn.Module, norm_type:float, clip_percentile:float):
        obs_grad_norm = self._get_grad_norm(model,norm_type)
        self.buffer.append(obs_grad_norm)
        clip_value = np.percentile(self.buffer, clip_percentile).item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value, norm_type)
    
    def __call__(self, model:nn.Module, norm_type:float=2, clip_percentile:float=10) -> None:
        return self._auto_clip_grad_norm(model,norm_type,clip_percentile)
    
    
class Save_helper:
    def __init__(self,
                 algo_name:str,
                 env_name="cstr_ode", 
                 seed=0):
        
        directory_name= 'PreTrained'
        
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        directory_name = directory_name + '/' + env_name + '/'
        
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
                
        directory_name = directory_name + algo_name + '/'

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        

        current_num_files = next(os.walk(directory_name))[1]

        run_num = len(current_num_files) + 1
        
        directory_name += "{}_{}".format(seed,run_num) + "/"
        
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
                
        self.directory_name = directory_name
        
        # ------ tensorboard set-up  ------#
        logdir = "tensorboard_experiments/"
        if not os.path.exists(logdir):
            os.makedirs(logdir)
            
        logdir = logdir + "{}_{}".format(algo_name, run_num)
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
            
        self.writer = SummaryWriter(log_dir=logdir)
        
    def model_path(self,episode_length):
        return self.directory_name + "{}.pth".format(episode_length)
    

    