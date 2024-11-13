import os
import numpy as np
import torch
from ignite.engine import EventEnum

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
    
def _get_grad_norm(model):
    with torch.no_grad():
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
    return total_norm 
 
# written for pytorch ignite, fire this on backwards pass
class BackwardsEvents(EventEnum):
    BACKWARDS_COMPLETED = 'backwards_completed'

def add_autoclip_gradient_handler(engine, model, clip_percentile):
    grad_history = []

    @engine.on(BackwardsEvents.BACKWARDS_COMPLETED)
    def autoclip_gradient(engine):
        obs_grad_norm = _get_grad_norm(model)
        grad_history.append(obs_grad_norm)
        clip_value = np.percentile(grad_history, clip_percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)